MY_DIR=`realpath $BASH_SOURCE | xargs dirname`
export PYTHONPATH=$MY_DIR
echo "added $MY_DIR to python path"