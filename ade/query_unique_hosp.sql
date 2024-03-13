use home_DLudwig1

-- import triplets.csv
-- 1048509 rows transferred 
-- select top 10 * from triplets

-- select * into triplets_yes from triplets where is_triplet = 1
-- 2414 rows affected

select count(*) from triplets_yes
-- 2414

select * from  information_schema.columns where table_name = 'triplets_yes'
--COLUMN_NAME
--Column_0
--deid_note_id
--PatientDurableKey
--HPI
--hosp_start
--hosp_end
--hosp_str
--med_start
--med_end
--med_str
--ae_start
--ae_end
--ae_str
--is_triplet
--med_str_generic

-- find the set of first mentions of any synonym set of hospitalizations by
-- grouping on deid_note_id, ae_start and finding the min hosp_start within group
drop table triplets_first_hosp0
select count(*) as first_hosp_cnt, deid_note_id, min(cast(hosp_start as int)) as min_hosp_start, ae_start
into triplets_first_hosp0
from triplets_yes
group by deid_note_id, ae_start
-- 722 rows

select * from triplets_first_hosp0 order by deid_note_id, min_hosp_start, ae_start

drop table triplets_first_hosp
select sum(first_hosp_cnt) as first_hosp_sum, deid_note_id, min_hosp_start, min(ae_start) as min_ae_start
into triplets_first_hosp
from triplets_first_hosp0
group by deid_note_id, min_hosp_start
-- 549 rows

select * from triplets_first_hosp order by deid_note_id, min_hosp_start, min_ae_start
