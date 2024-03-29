#!/bin/bash
#SBATCH -p akya-cuda        	# Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A ituzun         	# Kullanici adi
#SBATCH -J torch_experiment     # Gonderilen isin ismi
#SBATCH -o experiment.out    	# Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        	# Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                	# Gorev kac node'da calisacak?
#SBATCH -n 1                	# Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 10  	# Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=2:00:00      	# Sure siniri koyun.
#SBATCH --mail-user=e2036234@ceng.metu.edu.tr
#SBATCH --mail-type=ALL

eval "$(/truba/home/$USER/miniconda3/bin/conda shell.bash hook)"
conda activate torch-env
conda init

cd /truba/home/$USER/FSL_IDS/dashboard/FSL_CICIDS2017/
./run_experiments.sh new_tests/test_parameters_4-36-10__5__00.txt > run_experiments.log 2>&1
