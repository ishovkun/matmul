#include <iostream>

bool generate_matrices_and_multiply(int M, int N, int K);

auto main(int argc, char *argv[]) -> int {
  bool correct = true;
  correct = generate_matrices_and_multiply(16, 16, 16);
  if (!correct) return EXIT_FAILURE;
  correct = generate_matrices_and_multiply(64, 64, 64);
  if (!correct) return EXIT_FAILURE;
  correct = generate_matrices_and_multiply(1000, 1000, 1000);
  if (!correct) return EXIT_FAILURE;
  // correct = generate_matrices_and_multiply(2000, 2000, 2000);
  // if (!correct) return EXIT_FAILURE;

  return EXIT_SUCCESS;
}
