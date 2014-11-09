#!/bin/sh

TIMESTAMP=`date +%I%D | sed 's/\//_/g'`
LOG_DIR="/var/log"
APACHE_LOG_DIR="$LOG_DIR/apache2"
APACHE_ACCESS_LOG="$APACHE_LOG_DIR/access.log"
HOST_IP_TMP_FILE="/tmp/check_apache_$TIMESTAMP"

cleanup()
{
	echo "cleaning up"
	rm $HOST_IP_TMP_FILE*
}

collect_host_ips()
{
	echo "collecting host ips"
	cat /var/log/apache2/access.log | awk '{print $1}' | grep -v '::' | \
		sort > $HOST_IP_TMP_FILE.ip
}

collect_host_names()
{
	echo "collecting host names"
	for h in `cat $HOST_IP_TMP_FILE.ip`; do
		host $h >> $HOST_IP_TMP_FILE.name
	done;
}

make_histogram()
{
	echo "making occurance counts"
	awk ' { arr[$NF]++ } END { for( no in arr) { print no , arr[no] } } ' \
		$HOST_IP_TMP_FILE.name  | sort -k2 -r -n -o $HOST_IP_TMP_FILE.hist
}

##############
# PROCESSING #
##############
cleanup
collect_host_ips
collect_host_names
make_histogram



