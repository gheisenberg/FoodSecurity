<!DOCTYPE html>
<!-- 
//======================================================================
// FEMOZ PROJECT
// php script that fetches data from the PostgreSQL data base table and 
// renders it as a html table
//
// Author: 	Gernot Heisenberg
// Date: 	25.03.2022
//======================================================================
-->
<html>
 <head>
 	<title>FEMOZ Database</title>
	 <style>
			table {
				margin: 0 auto;
				font-size: large;
				border: 1px solid black;
			}
			h1 {
				text-align: center;
				color: #FF00C0;
				font-size: xx-large;
				font-family: 'Gill Sans', 'Gill Sans MT', 
				' Calibri', 'Trebuchet MS', 'sans-serif';
			}
			td {
				background-color: white;
				border: 1px solid black;
			}
			th,
			td {
				font-weight: bold;
				border: 1px solid black;
				padding: 10px;
				text-align: center;
			}
			td {
				font-weight: lighter;
			}
	 </style>
 </head>

  <body>

<?php 
	// DB connection parameters
	$host= '139.6.160.28';
	$db = 'raw_data_db';
	//$db = 'results_db';
	$user = 'gheisenberg';
	$password = 'kjaAH34!67Jse';
	
	try {
	$dsn = "pgsql:host=$host;port=5432;dbname=$db;";
	// establish a database connection
	$pdo = new PDO($dsn, $user, $password, [PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION]);

	// error handling
	/* if ($pdo) {
		print "Connected to the $db successfully!";
		}
	*/ 
	} 
	catch (PDOException $e) {
		die($e->getMessage());
		} 
	
	// construct the SQL query
	//$sqlquery = 'SELECT * FROM public.metadata ORDER BY Filename DESC';
	$sqlquery = 'SELECT * FROM public.metadata';

	$stmt   = $pdo -> query($sqlquery);
	$fetch = $stmt -> fetchAll(PDO::FETCH_ASSOC);
	//print_r ($fetch) . "\t";
	
	// finally close the connection
	$pdo = null;
	// end of php section
	?> 
	
	<section>
        <h1>Data sets in FEMOZ Database</h1>
        <!-- TABLE CONSTRUCTION-->
        <table>
            <tr>
                <th>filename</th>
                <th>title</th>
                <th>Contact person Femoz intern Email</th>
                <th>coverage district</th>
                <th>coverage province</th>
                <th>description</th>
            </tr>
            <!-- PHP CODE TO FETCH DATA FROM ROWS-->
            <?php   // LOOP TILL END OF DATA 
                //while($rows=$fetch->fetch_assoc())
				foreach($fetch as $rows)
                {
             ?>
            <tr>
                <!--FETCHING DATA FROM EACH 
                    ROW OF EVERY COLUMN-->
                <td><?php echo $rows['Filename'];?></td>
                <td><?php echo $rows['Title'];?></td>
                <td><?php echo $rows['Contact person Femoz intern Email'];?></td>
                <td><?php echo $rows['coverage district'];?></td>
                <td><?php echo $rows['coverage province'];?></td>
                <td><?php echo $rows['description'];?></td>
            </tr>
            <?php
                }
             ?>
        </table>
    </section>
   
  </body>
 </html>
