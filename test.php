<?php 

//echo "<h1>hello</h1><br>";

$waveForm = $_POST['waveform'];
$amplitude = $_POST['amplitude'];
$frequency = $_POST['frequency'];
$offset = $_POST['offset'];

//echo $waveForm . "</br>";
//echo $amplitude . "<br>"; 
//echo $frequency . "<br>"; 
//echo $offset . "<br>"; 
 
//$num = 5;

//$num ++;

//echo $num . "<br>";

if (strcmp($waveForm, "Sine") == 0){

	echo "ok";
}

//$name = array('liam',  'joe');
//echo $name[0] . "<br>";

//printf("Hi this is printf");


if($waveForm == "Sine"){
	
	$waveForm = '1';
	
}elseif($waveForm == "Square"){
	
	$waveForm = '2';
	
}elseif($waveForm == "Triangle"){
	
	$waveForm = '3';
}


$myfile = fopen("test.txt", "w") or die("Unable to open file!");


$txt = $waveForm.",".$amplitude.",".$frequency.",".$offset;
fwrite($myfile, $txt);

fclose($myfile);



?>
