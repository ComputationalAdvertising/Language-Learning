## C


### Socket

**Server端**

+ step1: 创建套接字；`socket`
+ step2: 绑定端口；`bind`
+ step3: 监听连接请求；`listen`
+ step4: 等待请求链接；`accept`

message放在`char buf[BUFFSIZE]`结构里。

**Client端**

+ step1: 创建client端套接字；`socket`
+ step2: 连接server socket；`connect`
+ step3: 接收server端消息；`recv`
+ step4: 向server端发送消息，&& 接收；`send/recv`


