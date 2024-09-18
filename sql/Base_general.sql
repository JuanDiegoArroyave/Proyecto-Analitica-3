---------------
------ 
---------------

drop table if exists base_general_modelo;

CREATE TABLE if not exists base_general_modelo AS
SELECT
*
FROM general_data1
WHERE SUBSTR(InfoDate, 1, 4) = '2015';

---------------
------ 
---------------

drop table if exists base_manager_survey;

CREATE TABLE if not exists base_manager_survey AS
SELECT
*
FROM manager_survey
WHERE SUBSTR(SurveyDate, 1, 4) = '2015';

---------------
------ 
---------------

drop table if exists base_employee_survey;

CREATE TABLE if not exists base_employee_survey AS
SELECT
*
FROM employee_survey_data
WHERE SUBSTR(DateSurvey, 1, 4) = '2015';

---------------
------ 
---------------

drop table if exists base_retirement;

CREATE TABLE if not exists base_retirement AS
SELECT
*
FROM retirement_info
WHERE SUBSTR(retirementDate, 1, 4) = '2016';




---------------
------ Se crea tabla base para el modelo, ésta contiene la información general de los empleados en 2015, 
------ junto a la encuesta de desempeño, encuesta de satisfacción y variable respuesta

---------------

drop table if exists tabla_general_modelo;

CREATE TABLE tabla_general_modelo AS
SELECT
t1.*,
t2.JobInvolvement,
t2.PerformanceRating,
t3.EnvironmentSatisfaction,
t3.JobSatisfaction,
t3.WorkLifeBalance,
t4.retirementDate,
t4.retirementType,
t4.resignationReason,
t4.Attrition

FROM base_general_modelo t1
LEFT JOIN base_manager_survey t2 on t1.EmployeeID = t2.EmployeeID
LEFT JOIN base_employee_survey t3 on t1.EmployeeID = t3.EmployeeID
LEFT JOIN base_retirement t4 ON t1.EmployeeID = t4.EmployeeID
