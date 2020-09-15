

epochs=("50")
batchsizes=("64")
model_choices=("lstm_attn" "lstm")
hidden_sizes_=("100 50")
seeds=("8687960" "456451" "12366")

for model_choice in "${model_choices[@]}" ; do
    for bs in "${batchsizes[@]}" ; do
        for hidden_sizes in "${hidden_sizes_[@]}" ; do
            for epoch in "${epochs[@]}" ; do
                for seed in "${seeds[@]}" ; do
                    python classifier.py --seed ${seed} --epochs ${epoch} --bs ${bs} --model ${model_choice} --hidden_sizes ${hidden_sizes}
                done
            done
        done
    done
done


epochs=("50")
batchsizes=("128")
model_choices=("lstm_attn")
hidden_sizes_=("100 50")
seeds=("8687960" "456451" "12366")


for model_choice in "${model_choices[@]}" ; do
    for bs in "${batchsizes[@]}" ; do
        for hidden_sizes in "${hidden_sizes_[@]}" ; do
            for epoch in "${epochs[@]}" ; do
                for seed in "${seeds[@]}" ; do
                    python classifier.py --seed ${seed} --epochs ${epoch} --bs ${bs} --model ${model_choice} --hidden_sizes ${hidden_sizes}
                done
            done
        done
    done
done

#epochs=("50")
#batchsizes=("64" "128")
#model_choices=("bi_lstm" "lstm" "rnn")
#hidden_sizes_=("50 20 10" "100 50")
#seeds=("8687960" "456451" "12366")

#epochs=("10" "20" "50")
#batchsizes=("64" "128")
#model_choices=("bi_lstm_attn" "lstm_attn" "bi_lstm" "lstm" "rnn")
#hidden_sizes_=("50 20" "100" "50" "50 20 10" "100 50")
#seeds=("8687960" "456451" "12366")
