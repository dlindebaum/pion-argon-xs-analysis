MY_DIR=`realpath $BASH_SOURCE | xargs dirname`
export MY_DIR
export PYTHONPATH=$MY_DIR
export PATH=$MY_DIR/apps:$PATH
export PATH=$MY_DIR/scripts:$PATH
echo "added $MY_DIR to python path"