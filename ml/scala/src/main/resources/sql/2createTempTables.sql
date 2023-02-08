--处理数据时所需要的临时表
use recommend;
CREATE TABLE IF NOT EXISTS tmp_dw_rcm_hitop_prepare2train_dm    
(
    device_id           STRING,
    label               STRING,
    hitop_id            STRING,
    screen              STRING,
    ch_name             STRING,
    author              STRING,
    sversion            STRING,
    mnc                 STRING,
    interface           STRING,
    designer            STRING,
    is_safe             INT,
    icon_count          INT,
    update_date         STRING,
    stars               DOUBLE,
    comment_num         INT,
    font                STRING,
    price               INT,
    file_size           INT,
    ischarge            SMALLINT,
    dlnum               INT,
    idlist              STRING,
    device_name         STRING,
    pay_ability         STRING
)row format delimited fields terminated by '\t';

--最终保存训练集的表
CREATE TABLE IF NOT EXISTS dw_rcm_hitop_prepare2train_dm 
(
    label                   STRING,
    features       STRING
)row format delimited fields terminated by '\t';
