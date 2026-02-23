# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

import cloudpickle

import vllm.envs as envs
from vllm.executor.ray_distributed_executor import (
    RayDistributedExecutor as RayDistributedExecutorV0,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.ray_utils import (
    FutureWrapper,
    RayWorkerWrapper,
    initialize_ray_cluster,
    ray,
)
from vllm.v1.outputs import ModelRunnerOutput

if ray is not None:
    from ray.actor import ActorHandle
else:
    ActorHandle = None

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

COMPLETED_NONE_FUTURE: Future[ModelRunnerOutput | None] = Future()
COMPLETED_NONE_FUTURE.set_result(None)


class RayDistributedExecutor(RayDistributedExecutorV0, Executor):
    """Ray-based distributed executor using V0 executor architecture for multi-node compatibility"""

    supports_pp: bool = True

    def _init_executor(self) -> None:
        # Call parent _init_executor which handles V0 initialization
        # This includes setting up use_ray_spmd_worker and use_ray_compiled_dag
        super()._init_executor()

        # KV connector setup
        self.has_connector = self.vllm_config.kv_transfer_config is not None

        self.uses_sampler = self.vllm_config.model_config.runner_type != "pooling" and (
            self.vllm_config.ec_transfer_config is None
            or not self.vllm_config.ec_transfer_config.is_ec_producer
        )

        self.scheduler_output: SchedulerOutput | None = None

    @property
    def max_concurrent_batches(self) -> int:
        """Ray distributed executor supports pipeline parallelism,
        meaning that it allows PP size batches to be executed concurrently.
        """
        pp_size = self.parallel_config.pipeline_parallel_size
        return 2 if pp_size <= 1 and self.scheduler_config.async_scheduling else pp_size

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        self.collective_rpc("reinitialize_distributed", args=(reconfig_request,))
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()

    def execute_model(  # type: ignore[override]
        self,
        scheduler_output: SchedulerOutput,
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        if self.scheduler_output is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        if not self.uses_sampler or not scheduler_output.total_num_scheduled_tokens:
            # Model will not execute, call model runner immediately.
            return self._execute_dag(scheduler_output, None, non_block)

        # Model will execute, defer to sample_tokens() call.
        self.scheduler_output = scheduler_output
        return COMPLETED_NONE_FUTURE if non_block else None

    def sample_tokens(  # type: ignore[override]
        self,
        grammar_output: "GrammarOutput | None",
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        """Execute the model on the Ray workers.

        The scheduler output to use should have been provided in
        a prior call to execute_model().

        Args:
            grammar_output: The structured outputs grammar bitmask, if applicable.
            non_block: If True, the method will return a Future.

        Returns:
            The model runner output.
        """
        scheduler_output = self.scheduler_output
        if scheduler_output is None:
            return COMPLETED_NONE_FUTURE if non_block else None

        self.scheduler_output = None

        return self._execute_dag(scheduler_output, grammar_output, non_block)

    def _execute_dag(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: "GrammarOutput | None",
        non_block: bool = False,
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        # Build the compiled DAG for the first time.
        if self.forward_dag is None:  # type: ignore
            self.forward_dag = self._compiled_ray_dag(enable_asyncio=False)

        refs = self.forward_dag.execute((scheduler_output, grammar_output))  # type: ignore

        if not self.has_connector:
            # Get output only from a single worker (output_rank)
            # When PP is not used, we block here until the result is available.
            if not non_block:
                return refs[0].get()

            # When PP is used, we return a FutureWrapper immediately so that
            # the scheduler can yield to the next batch.
            return FutureWrapper(refs[0])

        # Get output from all workers when connector is present
        assert self.kv_output_aggregator is not None
        if not non_block:
            # Block and get results from all workers
            return self.kv_output_aggregator.aggregate(ray.get(refs))

        # Return a future that will aggregate outputs from all workers
        return FutureWrapper(refs, self.kv_output_aggregator)

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
        non_block: bool = False,
    ) -> list[Any] | Future[list[Any]]:
        """Runs the given method on all workers."""
        sent_method = method if isinstance(method, str) else cloudpickle.dumps(method)
        del method

        if kwargs is None:
            kwargs = {}
        ray_worker_outputs = [
            worker.execute_method.remote(  # type: ignore[attr-defined]
                sent_method, *args, **kwargs
            )
            for worker in self.workers
        ]

        # Get the results of the ray workers.
        if non_block:
            return FutureWrapper(ray_worker_outputs)

        return ray.get(ray_worker_outputs, timeout=timeout)

    def _compiled_ray_dag(self, enable_asyncio: bool):
        assert self.parallel_config.use_ray
        self._check_ray_cgraph_installation()
        # Enlarge the default value of "RAY_CGRAPH_get_timeout" to 300 seconds
        # (it is 10 seconds by default). This is a Ray environment variable to
        # control the timeout of getting result from a compiled graph execution,
        # i.e., the distributed execution that includes model forward runs and
        # intermediate tensor communications, in the case of vllm.
        # Note: we should set this env var before importing
        # ray.dag, otherwise it will not take effect.
        os.environ.setdefault("RAY_CGRAPH_get_timeout", "300")  # noqa: SIM112
        from ray.dag import InputNode, MultiOutputNode

        logger.info(
            "RAY_CGRAPH_get_timeout is set to %s",
            os.environ["RAY_CGRAPH_get_timeout"],  # noqa: SIM112
        )
        logger.info(
            "VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE = %s",
            envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE,
        )
        logger.info(
            "VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM = %s",
            envs.VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM,
        )

        channel_type = envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
        if channel_type not in ("auto", "nccl", "shm"):
            raise ValueError(
                "Invalid value for VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE: "
                f"{channel_type}. Valid values are: 'auto', 'nccl', or 'shm'."
            )

        with InputNode() as input_data:
            # Example DAG: PP=2, TP=4
            #
            # SchedulerOutput -> 0 -> (SchedulerOutput, IntermediateTensors) -> 4 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 1 -> (SchedulerOutput, IntermediateTensors) -> 5 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 2 -> (SchedulerOutput, IntermediateTensors) -> 6 -> ModelRunnerOutput   # noqa: E501
            # SchedulerOutput -> 3 -> (SchedulerOutput, IntermediateTensors) -> 7 -> ModelRunnerOutput   # noqa: E501

            # All workers in the first TP group will take in the
            # ExecuteModelRequest as input.
            outputs = [input_data for _ in self.pp_tp_workers[0]]
            for pp_rank, tp_group in enumerate(self.pp_tp_workers):
                # Each PP worker takes in the output of the previous PP worker,
                # and the TP group executes in SPMD fashion.
                outputs = [
                    worker.execute_model_ray.bind(outputs[i])  # type: ignore[attr-defined]
                    for i, worker in enumerate(tp_group)
                ]

                last_pp_rank = len(self.pp_tp_workers) - 1
                if (
                    pp_rank < last_pp_rank
                    and envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE != "shm"
                ):
                    # Specify how intermediate tensors should be passed
                    # between pp stages, no need to specify for the last
                    # pp stage or when using shared memory (the default).
                    transport = envs.VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
                    outputs = [
                        output.with_tensor_transport(transport=transport)
                        for output in outputs
                    ]

            forward_dag = MultiOutputNode(outputs)

        if envs.VLLM_USE_RAY_WRAPPED_PP_COMM:
            from ray.experimental.channel.accelerator_context import (
                register_accelerator_context,
            )

            from vllm.distributed.device_communicators.ray_communicator import (
                RayPPCommunicator,
            )

            register_accelerator_context(
                torch_module_name="cuda", communicator_cls=RayPPCommunicator
            )
            logger.info(
                "Using RayPPCommunicator "
                "(which wraps vLLM _PP GroupCoordinator) "
                "for Ray Compiled Graph communication."
            )
        else:
            logger.info(
                "Using Ray's NCCL communicator for Ray Compiled Graph communication."
            )

        return forward_dag.experimental_compile(
            enable_asyncio=enable_asyncio,
            _overlap_gpu_communication=envs.VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM,
        )
