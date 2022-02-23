declare -a models=("lstm" "mlp" "cnn")
declare -a All_Round=(42 43 44 45 46)

for seed in "${All_Round[@]}"
do
    for model in "${models[@]}"
    do
        sbatch submit_training.sh ${seed} ${model} ${model}_vanilla
    done
done