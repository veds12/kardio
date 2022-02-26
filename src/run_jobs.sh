declare -a models=("cnn")
declare -a All_Round=(42 43 44 45 46)

for seed in "${All_Round[@]}"
do
    for model in "${models[@]}"
    do
        sbatch submit_training.sh ${seed} ${model} kardio_${model}_long
    done
done