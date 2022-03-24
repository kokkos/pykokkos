import os
import shutil
import pytest

# purge pk_cpp folder so that the
# test suite actually translates and
# compiles the code under test
cwd = os.getcwd()
shutil.rmtree(os.path.join(cwd, "pk_cpp"),
              ignore_errors=True)

# force pytest to actually import
# all the test modules directly
pytest.main()
