#include <stdio.h>
#include <Python.h>

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    printf("Usage: %s FILENAME\n", argv[0]);
    return 1;
  }
  FILE* cp = fopen(argv[1], "r");
  if (!cp)
  {
    printf("Error opening file: %s\n", argv[1]);
    return 1;
  }

  Py_Initialize();

  int rc = PyRun_SimpleFile(cp, argv[1]);
  fclose(cp);

  Py_Finalize();  
  return rc;
}