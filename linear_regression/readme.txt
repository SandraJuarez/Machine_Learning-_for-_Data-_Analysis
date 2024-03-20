Para el contenedor de docker:

>> docker run -d --rm --name mariadbcont -e MYSQL_ROOT_PASSWORD=root -p 3306:3306 mariadb

para correrlo en el bash
>>docker exec -it mariadbcont bash

necesitamos instalar git para bajar la base de datos

>> apt-get update
>> apt-get install git
>> git clone https://github.com/datacharmer/test_db.git

Importamos 

>> mysql < employees.sql -u root -proot