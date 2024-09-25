# OrganCMNIST Data
# random clustering + full cluster training
python ../GAT/src/main.py --clustering-method random --epoch 1000 --type full --save_root ../Results/GAT --save_folder OrganCRandom --save_name random_clustering

# random clustering + random sampling
python ../GAT/src/main.py --clustering-method random --epoch 1000 --save_root ../Results/GAT --type random_sampling --save_folder OrganCRandom --save_name random_clustering

# random clustering + diversity-driven sampling
python ../GAT/src/main.py --clustering-method random --epoch 1000 --save_root ../Results/GAT --type diversity_sampling --save_folder OrganCRandom --save_name random_clustering

# hyperedge clustering + full cluster training
python ../GAT/src/main.py --clustering-method random_hyper --epoch 1000 --type full --save_root ../Results/GAT --save_folder OrganCHyper --save_name hyperedge_clustering

# hyperedge clustering + random sampling
python ../GAT/src/main.py --clustering-method random_hyper --epoch 1000 --save_root ../Results/GAT --type random_sampling --save_folder OrganCHyper --save_name hyperedge_clustering

# hyperedge clustering + diversity-driven sampling
python ../GAT/src/main.py --clustering-method random_hyper --epoch 1000 --save_root ../Results/GAT  --type diversity_sampling --save_folder OrganCHyper --save_name hyperedge_clustering
