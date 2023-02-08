use recommend;
INSERT OVERWRITE TABLE tmp_dw_rcm_hitop_prepare2train_dm
SELECT
    t2.device_id,
    t2.label,
    t2.hitop_id,
    t2.screen,
    t2.ch_name,
    t2.author,
    t2.sversion,
    t2.mnc,
    t2.interface,
    t2.designer,
    t2.is_safe,
    t2.icon_count,
    to_date(t2.update_time),
    t2.stars,
    t2.comment_num,
    t2.font,
    t2.price,
    t2.file_size,
    t2.ischarge,
    t2.dlnum,
    t1.devid_applist,
    t1.device_name,
    t1.pay_ability
FROM
	(
		SELECT
			device_id,
			devid_applist,
			device_name,
			pay_ability
		FROM
			dw_rcm_hitop_userapps_dm
	) t1
RIGHT OUTER JOIN 
	(
		SELECT
			device_id,
			label,
			hitop_id,
			screen,
			ch_name,
			author,
			sversion,
			IF (mnc IN ('00','01','02','03','04','05','06','07'), mnc,'x')      AS   mnc,
			interface,
			designer,
			is_safe,
			IF (icon_count <= 5,icon_count,6)                                   AS   icon_count,
			update_time,
			stars,
			IF ( comment_num IS NULL,0,IF ( comment_num <= 10,comment_num,11))  AS   comment_num,
			font,
			price,
			IF (file_size <= 2*1024*1024,2,
			IF (file_size <= 4*1024*1024,4,
			IF (file_size <= 6*1024*1024,6,
			IF (file_size <= 8*1024*1024,8,
			IF (file_size <= 10*1024*1024,10,
			IF (file_size <= 12*1024*1024,12,
			IF (file_size <= 14*1024*1024,14,
			IF (file_size <= 16*1024*1024,16,
			IF (file_size <= 18*1024*1024,18,
			IF (file_size <= 20*1024*1024,20,21))))))))))    AS    file_size,
			ischarge,
			IF (dlnum IS NULL,0,
			IF (dlnum <= 50,50,
			IF (dlnum <= 100,100,
			IF (dlnum <= 500,500,
			IF (dlnum <= 1000,1000,
			IF (dlnum <= 5000,5000,
			IF (dlnum <= 10000,10000,
			IF (dlnum <= 20000,20000,20001))))))))          AS      dlnum
		FROM
			dw_rcm_hitop_sample2learn_dm
	) t2
ON (t1.device_id = t2.device_id);
