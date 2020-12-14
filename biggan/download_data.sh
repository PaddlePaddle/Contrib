#download cifar10 data
wget -O cifar-10.tar.gz "https://bj.bcebos.com/v1/ai-studio-online/2fd8e8506ca14c5f91860b001294da9240d7f5e970384fa0a7bef457c874ea03?responseContentDisposition=attachment%3B%20filename%3Dcifar-10.tar.gz"
tar xzfv cifar-10.tar.gz
mkdir -p BigGAN-paddle/data/cifar
#move it to the biggan data folder
mv cifar-10-batches-py BigGAN-paddle/data/cifar/



#download inception pretrain model for evaluation metric FID, and inception score
wget -O inception_model.pdparams "https://bj.bcebos.com/v1/ai-studio-online/c43f3966d78a4c55845e57e041c9bdc8e2507ac8bd6548dd838429003e8e74ec?responseContentDisposition=attachment%3B%20filename%3Dinception_model.pdparams"
mv inception_model.pdparams BigGAN-paddle/ 

