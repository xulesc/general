#!/bin/sh

BASE="BASE_DIR"
TODO="SPACE_SEPARATED_DIR_NAMES"
RURL="https://github.com/USER_NAME"

for todo in $TODO; do
	echo "doing: $todo"
	cp -a $BASE $todo
	cd $todo
	git filter-branch --prune-empty --subdirectory-filter $todo master
	git remote set-url origin $RURL/$todo.git
	git push
	cd ..
	rm -rf $todo
done;
