## Bash utility function for various tasks

## Create n-fold split of dataset with files in a folder for the purpose of classifier
## evaluation. The function expects path to dataset as first parameter and the number
## of folds to create as second parameter. The list of files which contain names of the
## files belonging are written to a variable $fold_files
nfold_split()
{
	data_set_dir=$1
	n_fold=$2
	##
	ls $data_set_dir > /tmp/names
	sort -R /tmp/names > /tmp/data_files
	split --number=$n_fold /tmp/data_files /tmp/fold
	rm /tmp/names
	rm /tmp/data_files
	##
	fold_files=$(ls /tmp/fold*)
}

##