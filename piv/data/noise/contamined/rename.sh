for file in `ls | grep .png`
do
  newfile=`echo $file | cut -c 3-`
  mv $file $newfile
done
for file in `ls | grep .pdparams`
do
  newfile=`echo $file | cut -c 3-`
  mv $file $newfile
done
