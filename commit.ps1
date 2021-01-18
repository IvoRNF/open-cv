param ($msg) 

if(-not ($msg)){
   $msg = '...'
}

& git add . 
& git commit --m $msg
& git push origin master