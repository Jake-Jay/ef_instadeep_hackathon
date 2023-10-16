# download .gz archives
mkdir tmp; cd tmp
./oas_gupta.sh
gzip -d *.gz
cd ..
# extract .csv files
mkdir ../Gupta_2017
mv tmp/*.gz ../Gupta_2017 
