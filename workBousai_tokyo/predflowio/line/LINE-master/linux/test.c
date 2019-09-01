#include <stdio.h>
#include <gsl/gsl_sf_bessel.h>
int main(void)
{
  double x = 5.0;
  printf("hello\n");
  double y = gsl_sf_bessel_J0(x);
  printf("J0(%g) = %.18e\n", x, y);
#ifdef WIN32
  system("pause");
#endif
  return 0;
}