CUDA_VISIBLE_DEVICES='0,1,2,3' python3 robust_self_training_GAIR.py --aux_data_filename='ti_500K_pseudo_labeled.pickle' --distance='l_inf' --epsilon=0.031 --Lambda=3.0 --model_dir='./GAIR_RST_cw_100_3' &

wait

echo "ALL DONE!"