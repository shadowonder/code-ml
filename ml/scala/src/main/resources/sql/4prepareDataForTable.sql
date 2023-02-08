--添加python处理脚本
ADD FILE /root/test/dw_rcm_hitop_prepare2train_dm.py;

use recommend;

-- 向训练集表dw_rcm_hitop_prepare2train_dm中插入数据
-- 这里将字段丢给python文件去处理
INSERT OVERWRITE TABLE dw_rcm_hitop_prepare2train_dm
SELECT
TRANSFORM (t.*)
USING 'python dw_rcm_hitop_prepare2train_dm.py'
AS (label,features)
FROM
(
    SELECT 
        label,
        hitop_id,
        screen,
        ch_name,
        author,
        sversion,
        mnc,
        interface,
        designer,
        icon_count,
        update_date,
        stars,
        comment_num,
        font,
        price,
        file_size,
        ischarge,
        dlnum,
        idlist,
        device_name,
        pay_ability
    FROM 
        tmp_dw_rcm_hitop_prepare2train_dm
) t;
