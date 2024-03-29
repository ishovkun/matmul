#include <iostream>
#include <thrust/device_vector.h>

enum class Method
{
  naive,
  tiled1,
  tiled2
};

__global__ void matmul_kernel(double const *A, double const *B, double *res,
                              int M, int N, int K)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < M*N) {
    int rowA = tid / N;
    int colB = tid % K;
    double tmp = 0;
    for (int i = 0 ; i < N; i++) {
      tmp += A[rowA*N + i] * B[i*K + colB];
    }
    res[rowA*K + colB] = tmp;
  }
}

__global__ void matmul_kernel_tiled(double const *A, double const *B, double *C, int M, int N, int K)
{
  __shared__ double smem;

  int tile_size = blockDim.x * blockDim.y;
  int tid = threadIdx.x + threadIdx.y*blockDim.x;  // index within block

  // int TM = (M+tile_size-1) / tile_size; //  num row tiles in A
  int TN = (N+tile_size-1) / tile_size;  // num tiles in N direction
  int TK = (K+tile_size-1) / tile_size;  // num tiles in K directions

  int row_tile_A = blockIdx.x / TN;
  int col_tile_A = blockIdx.x % TN;


  int row_tile_B = col_tile_A;
  // int row_tile_B = blockIdx.y / TK;
  int col_tile_B = blockIdx.y;

  if (tid == 0) {
    smem = 0;
  }

  // if (tid == 0) {
  //   printf("tileA = [%d, %d] tileB = [%d, %d] block = [%d, %d]\n",
  //          row_tile_A, col_tile_A,
  //          row_tile_B, col_tile_B,
  //          blockIdx.x, blockIdx.y);
  // }

  for (int colB = col_tile_B*tile_size; colB < min(N, (col_tile_B+1)*tile_size); colB++) {
    int rowB = row_tile_B*tile_size + tid;
    double b = B[rowB*K + colB];

    for (int rowA = row_tile_A*tile_size; rowA < min(M, (row_tile_A+1)*tile_size); rowA++) {

      // save into shared
      int colA = col_tile_A*tile_size + tid;
      double a = A[rowA*M + colA];
      double c = a*b;
      atomicAdd(&smem, c);

      // write into global
      __syncthreads();
      if (tid == 0) {
        // printf("A[%d, %d] += %f tiles [%d,%d] * [%d,%d]\n", rowA, colB, c,
        //    row_tile_A, col_tile_A,
        //    row_tile_B, col_tile_B);
        atomicAdd(&C[rowA*K + colB], smem);
        smem = 0;
      }
    }
  }
}

float matmul(double const *A, double const *B, double *res, int M, int N, int K, Method method = Method::naive)
{
  thrust::fill(thrust::device_ptr<double>(res), thrust::device_ptr<double>(res) + M * K, 0);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (method == Method::naive) {
    int nt = 256;
    int nb = (M * K + nt - 1) / nt;
    cudaEventRecord(start);
    matmul_kernel<<<nb, nt>>>(A, B, res, M, N, K);
    cudaEventRecord(stop);
  }
  else if (method == Method::tiled1) {
    unsigned int tile_size = 4;
    dim3 nt {tile_size, 1, 1};
    int ntM = (M+tile_size-1)/tile_size;
    int ntN = (N+tile_size-1)/tile_size;
    int ntK = (K+tile_size-1)/tile_size;
    dim3 nb( ntM * ntN, ntK, 1);
    std::cout << "nb = " << nb.x << " x " << nb.y  << std::endl;


    cudaEventRecord(start);
    matmul_kernel_tiled<<<nb,nt>>>(A, B, res, M, N, K);
    cudaEventRecord(stop);
  }

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  return milliseconds;
}

template<typename T>
bool compare(thrust::device_vector<T> const & v1, thrust::device_vector<T> const & v2, double tol = 1e-15)
{
  size_t num_errors = 0;
  for (size_t i = 0; i < v1.size(); ++i) {
    auto diff = std::abs(v1[i] - v2[i]);
    auto cmp = std::max((decltype(diff))1, std::max(std::abs(v1[i]), std::abs(v2[i])));
    if ( diff > tol*cmp || std::isnan(diff) || std::isinf(diff)) {
      std::cout<< i << "  " << v1[i] << " " << v2[i] << " diff = " << diff << " > " << (tol*cmp) << std::endl;
      num_errors++;
    }
    if ( num_errors > 10 ) break;
  }
  return (num_errors == 0);
}

void generate_matrices_and_multiply(int M, int N, int K)
{
  thrust::device_vector<double> A(M*N);
  thrust::device_vector<double> B(N*K);
  thrust::device_vector<double> C_ref(M*K);
  thrust::device_vector<double> C(M*K);

  thrust::fill(A.begin(), A.end(), 1);
  thrust::fill(B.begin(), B.end(), 2);

  auto t1 = matmul(A.data().get(), B.data().get(), C_ref.data().get(), M, N, K, Method::naive);
  auto t2 = matmul(A.data().get(), B.data().get(), C.data().get(), M, N, K, Method::tiled1);
  compare(C_ref, C);
  // auto t3 = matmul_tiled1(A.data().get(), B.data().get(), C.data().get(), M, N, K);
  // compare(C_ref, C);

  std::cout << "t1 = " << t1 << std::endl;
  std::cout << "t2 = " << t2 << std::endl;
  // std::cout << "t3 = " << t3 << std::endl;



  // for (int row = 0; row < M; row++) {
  //   for (int col = 0; col < K; col++) {
  //     std::cout << C[row * K + col] << " ";
  //   }
  //   std::cout << std::endl;
  // }

}
