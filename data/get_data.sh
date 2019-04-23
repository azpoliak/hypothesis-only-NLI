mkdir rte
cd rte
wget https://github.com/decompositional-semantics-initiative/DNC/raw/master/inference_is_everything.zip
unzip inference_is_everything.zip
rm inference_is_everything.zip
cd ../
#echo "About to split the data into formats for train.lua and eval.lua"
python convert_recast_white.py

echo "Downloading SNLI"
wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
unzip snli_1.0.zip

echo "Reformatting SNLI dataset"
python convert_snli.py

echo "Downloading GloVe"
mkdir embds
cd embds
curl -LO http://nlp.stanford.edu/data/glove.840B.300d.zip
jar xvf glove.840B.300d.zip 
#rm glove.840B.300d.zip

#echo "Downloading multi-SNLI"
#wget http://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip
#unzip multinli_1.0

#echo "Reformatting Multi-NLI dataset"
#python convert_mnli.py

#echo "Downloading Compositional NLI"
#mdkir compositional-rte
#cd compositional-rte
#svn export https://github.com/ishita-dg/ScrambleTests/trunk/testData/

echo "Downloading MPE"
mkdir mpe
curl https://raw.githubusercontent.com/aylai/MultiPremiseEntailment/master/data/MPE/mpe_train.txt -o mpe/mpe_train.txt
curl https://raw.githubusercontent.com/aylai/MultiPremiseEntailment/master/data/MPE/mpe_dev.txt -o mpe/mpe_dev.txt
curl https://raw.githubusercontent.com/aylai/MultiPremiseEntailment/master/data/MPE/mpe_test.txt -o mpe/mpe_test.txt

echo "Downloading add-1 RTE"
mkdir add-one-rte
cd add-one-rte
wget http://www.seas.upenn.edu/~nlp/resources/AN-composition.tgz
tar -zxvf AN-composition.tgz 
rm AN-composition.tgz 

echo "Downloading SICK"
mkdir sick
cd sick
wget http://clic.cimec.unitn.it/composes/materials/SICK.zip
unzip SICK.zip
rm SICK.zip
cd ../
python convert_sick.py

echo "Downloading SciTail"
mkdir scitail
cd scitail
wget http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.zip
unzip SciTailV1.zip
rm SciTailV1.zip
cd ../
python convert_scitail.py

echo "Downloading JOCI"
echo "Need to fix extracting JOCI"
mkdir joci
cd joci
wget http://decomp.io/projects/common-sense-inference/joci-A.zip
wget http://decomp.io/projects/common-sense-inference/joci-B.zip
unzip joci.zip
cd data
tar -zxvf joci.tar.gz
