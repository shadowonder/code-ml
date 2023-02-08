use recommend;
--创建商品词表 dim_rcm_hitop_id_list_ds 
CREATE EXTERNAL TABLE IF NOT EXISTS dim_rcm_hitop_id_list_ds
(
    hitop_id    STRING,
    name        STRING,
    author      STRING,
    sversion    STRING,
    ischarge    SMALLINT,
    designer    STRING,
    font        STRING,
    icon_count  INT,
    stars       DOUBLE,
    price       INT,
    file_size   INT,     
    comment_num INT,
    screen      STRING,
    dlnum       INT
)row format delimited fields terminated by '\t';
load data local inpath '/root/test/applist.txt' into table dim_rcm_hitop_id_list_ds;

--创建用户历史下载表	dw_rcm_hitop_userapps_dm
CREATE EXTERNAL TABLE IF NOT EXISTS dw_rcm_hitop_userapps_dm
(
    device_id           STRING,
    devid_applist       STRING,
    device_name         STRING,
    pay_ability         STRING
)row format delimited fields terminated by '\t';
load data local inpath '/root/test/userdownload.txt' into table dw_rcm_hitop_userapps_dm;

--创建正负例样本表 dw_rcm_hitop_sample2learn_dm
CREATE EXTERNAL TABLE IF NOT EXISTS dw_rcm_hitop_sample2learn_dm 
(
    label       STRING,
    device_id   STRING,
    hitop_id    STRING,
    screen      STRING,
    en_name     STRING,
    ch_name     STRING,
    author      STRING,
    sversion    STRING,
    mnc         STRING,
    event_local_time STRING,
    interface   STRING,
    designer    STRING,
    is_safe     INT,
    icon_count  INT,
    update_time STRING,
    stars       DOUBLE,
    comment_num INT,
    font        STRING,
    price       INT,
    file_size   INT,
    ischarge    SMALLINT,
    dlnum       INT
)row format delimited fields terminated by '\t';
load data local inpath '/root/test/sample.txt' into table dw_rcm_hitop_sample2learn_dm;