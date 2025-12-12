MY_DIR=`(realpath $BASH_SOURCE | xargs dirname) || pwd`
export MY_DIR
export PYTHONPATH=$MY_DIR
export CONFIG_PATH=$MY_DIR/config
export PATH=$MY_DIR/apps:$PATH
export PATH=$MY_DIR/scripts:$PATH
echo "added $MY_DIR to python path"