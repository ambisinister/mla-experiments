cd ./data/
python download_data.py
python hftokenizer.py
python construct_dataset.py
cp -R hftokenizer ../
cd ../
mkdir ./figures
mkdir ./weights

python train_model.py
