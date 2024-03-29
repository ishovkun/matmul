#include <iostream>
#include <thrust/device_vector.h>

enum class Method
{
  naive,
  tiled1,
  tiled2,
  tiled3,
  tiled4,
  tiled5,
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

  int TN = (N+tile_size-1) / tile_size;  // num tiles in N direction

  int row_tile_A = blockIdx.x / TN;
  int col_tile_A = blockIdx.x % TN;
  int row_tile_B = col_tile_A;
  int col_tile_B = blockIdx.y;

  if (tid == 0) {
    smem = 0;
  }

  for (int colB = col_tile_B*tile_size; colB < min(N, (col_tile_B+1)*tile_size); colB++) {
    int rowB = row_tile_B*tile_size + tid;
    if (rowB >= N) return;

    double b = B[rowB*K + colB];

    for (int rowA = row_tile_A*tile_size; rowA < min(M, (row_tile_A+1)*tile_size); rowA++) {

      // save into shared
      int colA = col_tile_A*tile_size + tid;
      if (colA >= N) return;
      double a = A[rowA*M + colA];
      double c = a*b;
      atomicAdd(&smem, c);

      // write into global
      __syncthreads();
      if (tid == 0) {
        atomicAdd(&C[rowA*K + colB], smem);
        smem = 0;
      }
    }
  }
}

__global__ void matmul_kernel_tiled2(double const *A, double const *B, double *C, int M, int N, int K)
{
  extern __shared__ double smem[];

  int tile_size = blockDim.x * blockDim.y;
  int tid = threadIdx.x + threadIdx.y*blockDim.x;  // index within block

  int TN = (N+tile_size-1) / tile_size;  // num tiles in N direction

  int row_tile_A = blockIdx.x / TN;
  int col_tile_A = blockIdx.x % TN;
  int row_tile_B = col_tile_A;
  int col_tile_B = blockIdx.y;

  for (int colB = col_tile_B*tile_size; colB < min(N, (col_tile_B+1)*tile_size); colB++) {
    int rowB = row_tile_B*tile_size + tid;
    if (rowB >= N) return;

    double b = B[rowB*K + colB];

    for (int rowA = row_tile_A*tile_size; rowA < min(M, (row_tile_A+1)*tile_size); rowA++) {

      // save into shared
      int colA = col_tile_A*tile_size + tid;
      if (colA >= N) return;
      double a = A[rowA*M + colA];
      double c = a*b;
      smem[tid] = c;

      // write into global
      __syncthreads();
      if (tid == 0) {
        double ans = 0;
        for (int i = 0; i < tile_size; i++)
          ans += smem[i];
        atomicAdd(&C[rowA*K + colB], ans);
      }
    }
  }
}

__global__ void matmul_kernel_tiled3(double const *A, double const *B, double *C, int M, int N, int K)
{
  extern __shared__ double smem[];

  int tile_size = blockDim.x * blockDim.y;
  int tid = threadIdx.x + threadIdx.y*blockDim.x;  // index within block

  int TN = (N+tile_size-1) / tile_size;  // num tiles in N direction

  int row_tile_A = blockIdx.x / TN;
  int col_tile_A = blockIdx.x % TN;
  int row_tile_B = col_tile_A;
  int col_tile_B = blockIdx.y;

  for (int colB = col_tile_B*tile_size; colB < min(N, (col_tile_B+1)*tile_size); colB++) {
    int rowB = row_tile_B*tile_size + tid;
    if (rowB >= N) return;

    double b = B[rowB*K + colB];

    for (int rowA = row_tile_A*tile_size; rowA < min(M, (row_tile_A+1)*tile_size); rowA++) {

      // save into shared
      int colA = col_tile_A*tile_size + tid;
      if (colA >= N) return;
      double a = A[rowA*M + colA];
      double c = a*b;
      int rowS = rowA - row_tile_A*tile_size;
      smem[rowS*tile_size + tid] = c;

      // write into global
    }
    __syncthreads();

    double ans = 0;
    for (int i = 0; i < tile_size; i++)
      ans += smem[tid*tile_size + i];

    int rowA = row_tile_A*tile_size + tid;
    atomicAdd(&C[rowA*K + colB], ans);

    // if (tid == 0) {
    //   double ans = 0;
    //   for (int i = 0; i < tile_size; i++)
    //     ans += smem[i];
    //   atomicAdd(&C[rowA*K + colB], ans);
    // }

  }
}

__global__ void matmul_kernel_tiled4(double const *A, double const *B, double *C, int M, int N, int K)
{
  extern __shared__ double smem[];
  int tile_size = blockDim.x;
  double* sA = &smem[0];
  double* sB = &smem[tile_size*tile_size];

  int TN = (N+tile_size-1) / tile_size;  // num tiles in N direction
  int row_tile_A = blockIdx.x / TN;
  int col_tile_A = blockIdx.x % TN;
  int row_tile_B = col_tile_A;
  int col_tile_B = blockIdx.y;

  int rowA = row_tile_A*tile_size + threadIdx.x;
  int colA = col_tile_A*tile_size + threadIdx.y;
  sA[threadIdx.x*tile_size + threadIdx.y] = (rowA < M && colA < N) ? A[ rowA*N + colA ] : 0;
  int rowB = row_tile_B*tile_size + threadIdx.x;
  int colB = col_tile_B*tile_size + threadIdx.y;
  sB[threadIdx.x*tile_size + threadIdx.y] = (rowB < N && colB < K) ? B[ rowB*K + colB ] : 0;
  __syncthreads();

  double tmp = 0;
  for (int k = 0; k < tile_size; k++) {
    tmp += sA[threadIdx.y * tile_size + k] * sB[k * tile_size + threadIdx.x];
  }

  if (rowA < N && colB < K) {
    atomicAdd(&C[rowA * K + colB], tmp);
  }
}

__global__ void matmul_kernel_tiled5(double const *A, double const *B, double *C, int M, int N, int K)
{
  extern __shared__ double smem[];
  int tile_size = blockDim.x;
  double *sA = &smem[0];
  double *sB = &smem[tile_size*tile_size];

  int nctA = (N + tile_size - 1) / tile_size;
  int rtA = blockIdx.x;
  int rA = rtA*tile_size + threadIdx.x;
  int ctB = blockIdx.y;
  int cB = ctB*tile_size + threadIdx.y;

  for ( int ctA = 0; ctA < nctA; ctA++) {

    // load A
    int cA = ctA*tile_size + threadIdx.y;
    sA[threadIdx.x * tile_size + threadIdx.y] = (rA < M && cA < N) ? A[rA*N + cA] : 0;
    int rtB = ctA;
    int rB = rtB*tile_size + threadIdx.x;
    // load B
    sB[threadIdx.x * tile_size + threadIdx.y] = (rB < N && cB < N) ? B[rB*K + cB] : 0;
    __syncthreads();

    // do the product
    double tmp{0};
    for (int i = 0; i < tile_size; i++)
      tmp += sA[threadIdx.x*tile_size + i] * sB[i*tile_size + threadIdx.y];

    // atomicAdd(&C[rA*K + cB], tmp);
    C[rA*K + cB] += tmp;
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
    unsigned int tile_size = 32;
    dim3 nt {tile_size, 1, 1};
    int ntM = (M+tile_size-1)/tile_size;
    int ntN = (N+tile_size-1)/tile_size;
    int ntK = (K+tile_size-1)/tile_size;
    dim3 nb( ntM * ntN, ntK, 1);
    cudaEventRecord(start);
    matmul_kernel_tiled<<<nb,nt>>>(A, B, res, M, N, K);
    cudaEventRecord(stop);
  }
  else if (method == Method::tiled2) {
    unsigned int tile_size = 32;
    dim3 nt {tile_size, 1, 1};
    int ntM = (M+tile_size-1)/tile_size;
    int ntN = (N+tile_size-1)/tile_size;
    int ntK = (K+tile_size-1)/tile_size;
    dim3 nb( ntM * ntN, ntK, 1);
    cudaEventRecord(start);
    matmul_kernel_tiled2<<<nb,nt,tile_size>>>(A, B, res, M, N, K);
    cudaEventRecord(stop);
  }
  else if (method == Method::tiled4) {
    unsigned int tile_size = 8;
    dim3 nt {tile_size, tile_size, 1};
    int ntM = (M+tile_size-1)/tile_size;
    int ntN = (N+tile_size-1)/tile_size;
    int ntK = (K+tile_size-1)/tile_size;
    dim3 nb( ntM * ntN, ntK, 1);
    cudaEventRecord(start);
    matmul_kernel_tiled4<<<nb,nt,2*tile_size*tile_size*sizeof(double)>>>(A, B, res, M, N, K);
    cudaEventRecord(stop);
  }
  else if (method == Method::tiled5) {
    unsigned int tile_size = 16;
    dim3 nt {tile_size, tile_size, 1};
    int ntN = (N+tile_size-1)/tile_size;
    int ntK = (K+tile_size-1)/tile_size;
    dim3 nb( ntN, ntK, 1);
    cudaEventRecord(start);
    matmul_kernel_tiled5<<<nb,nt,2*tile_size*tile_size*sizeof(double)>>>(A, B, res, M, N, K);
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

bool generate_matrices_and_multiply(int M, int N, int K)
{
  thrust::device_vector<double> A(M*N);
  thrust::device_vector<double> B(N*K);
  thrust::device_vector<double> C_ref(M*K);
  thrust::device_vector<double> C(M*K);

  thrust::fill(A.begin(), A.end(), 1);
  thrust::fill(B.begin(), B.end(), 2);

  std::vector<float> timings;

  timings.push_back(matmul(A.data().get(), B.data().get(), C_ref.data().get(), M, N, K, Method::naive) );
  timings.push_back(matmul(A.data().get(), B.data().get(), C.data().get(), M, N, K, Method::tiled1));
  if (M*N*K < 1e6 && !compare(C_ref, C)) return false;
  // timings.push_back(matmul(A.data().get(), B.data().get(), C.data().get(), M, N, K, Method::tiled2));
  timings.push_back(matmul(A.data().get(), B.data().get(), C.data().get(), M, N, K, Method::tiled4));
  if (M*N*K < 1e6 && !compare(C_ref, C)) return false;
  timings.push_back(matmul(A.data().get(), B.data().get(), C.data().get(), M, N, K, Method::tiled5));
  if (M*N*K < 1e6 && !compare(C_ref, C)) return false;
  if (M*N*K >= 1e6) {
    for (int i = 0; i < timings.size(); i++) {
      std::cout << "timing[" << i << "] = " << timings[i] << std::endl;
    }
  }
  return true;
}
