#include <stdio.h>
int main(int argc, char** argv)
{
  FILE *out;
  out = fopen("test", "w");
  fprintf(out, "hahahaha");
  return 0;
}
