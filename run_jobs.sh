declare -a models=("lstm" "cnn" "mlp")


for model in "${models[@]}"
do
    nohup python train.py --model ${model} --logging 1 --verbose 1 --name ${model}_vanilla --checkpoint ./checkpoint --epochs 75 > nohup/${model}_vanilla.out
done