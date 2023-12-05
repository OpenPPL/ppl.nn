#include <vector>
#include <iostream>

#include <omp.h>

#include <cuda_runtime.h>
#include <nccl.h>

const int num_gpus = 2;

#define NCCL_CHECK(cmd, emsg)                                                \
    do {                                                                     \
        ncclResult_t e = (cmd);                                              \
        if (e != ncclSuccess) {                                              \
            std::cerr << "NCCL error(code:" << (int)e << ") on " << (emsg) << std::endl; \
            exit(-1);                                                        \
        }                                                                    \
    } while (0);

#define CUDA_CHECK(cmd, emsg)                                                \
    do {                                                                     \
        cudaError_t e = (cmd);                                              \
        if (e != cudaSuccess) {                                              \
            std::cerr << "CUDA error(code:" << (int)e << ") on " << (emsg) << std::endl; \
            exit(-1);                                                        \
        }                                                                    \
    } while (0);

static void NcclAllReduceSumHalf(const void* send_buf, void* recv_buf, const int data_size, ncclComm_t comm, cudaStream_t stream)
{
    NCCL_CHECK(ncclGroupStart(), "ncclGroupStart");
    NCCL_CHECK(ncclAllReduce(send_buf, recv_buf, data_size, ncclHalf, ncclSum, comm, stream), "ncclAllReduce");
    NCCL_CHECK(ncclGroupEnd(), "ncclGroupEnd");
}

static void NcclAllGatherHalf(const void* send_buf, void* recv_buf, const int data_size, ncclComm_t comm, cudaStream_t stream)
{
    NCCL_CHECK(ncclGroupStart(), "ncclGroupStart");
    NCCL_CHECK(ncclAllGather(send_buf, recv_buf, data_size, ncclHalf, comm, stream), "ncclAllGather");
    NCCL_CHECK(ncclGroupEnd(), "ncclGroupEnd");
}

static void InitCudaThread() {
    #pragma omp parallel num_threads(num_gpus)
    {
        auto tid = omp_get_thread_num();
        CUDA_CHECK(cudaSetDevice(tid), "cudaSetDevice");
    }
}

static void InitNccl(ncclComm_t *comms) {
    ncclUniqueId uuid;
    NCCL_CHECK(ncclGetUniqueId(&uuid), "ncclGetUniqueId");
    #pragma omp parallel num_threads(num_gpus)
    {
        auto tid = omp_get_thread_num();
        NCCL_CHECK(ncclCommInitRank(&comms[tid], num_gpus, uuid, tid), "ncclCommInitRank");
    }
}

static void FinalizeNccl(ncclComm_t *comms) {
    #pragma omp parallel num_threads(num_gpus)
    {
        auto tid = omp_get_thread_num();
        NCCL_CHECK(ncclCommDestroy(comms[tid]), "ncclCommDestroy");
    }
}

static void RunAllReduceHalf(int num_elements, ncclComm_t *comms) {
    #pragma omp parallel num_threads(num_gpus)
    {
        auto tid = omp_get_thread_num();

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

        void *dmem = nullptr;
        void *hmem = malloc(num_elements * sizeof(short));
        CUDA_CHECK(cudaMallocAsync(&dmem, num_elements * sizeof(short), stream), "cudaMallocAsync");
        CUDA_CHECK(cudaMemsetAsync(dmem, 0, num_elements * sizeof(short), stream), "cudaMemsetAsync");
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        NcclAllReduceSumHalf(dmem, dmem, num_elements, comms[tid], stream);
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

        CUDA_CHECK(cudaMemcpyAsync(hmem, dmem, num_elements * sizeof(short), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync");

        if (tid == 1)
            std::cout << "[    OK] " << *(short*)hmem << std::endl;

        CUDA_CHECK(cudaFreeAsync(dmem, stream), "cudaFreeAsync");
        CUDA_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        CUDA_CHECK(cudaStreamDestroy(stream), "cudaStreamDestroy");

        free(hmem);
    }
}

int main(int argc, char* argv[]) {
    std::vector<ncclComm_t> comms(num_gpus);
    InitCudaThread();
    InitNccl(comms.data());

    for (int i = 4 * 1024; i <= 512 * 1024 * 1024; i <<= 1) {
        std::cout << "[RUN   ] AllReduce " << i / 1024 << " KB" << std::endl;
        RunAllReduceHalf(i / sizeof(short), comms.data());
    }

    FinalizeNccl(comms.data());

    return 0;
}
