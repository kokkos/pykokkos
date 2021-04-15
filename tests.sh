readonly _DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
readonly TESTS_DIR="${_DIR}"/tests
readonly RC_FILE="${_DIR}"/.coveragerc

coverage run --rcfile="${RC_FILE}" -m unittest discover -s "${TESTS_DIR}"
coverage report
coverage xml -o cov.xml