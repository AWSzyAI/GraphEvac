加入冗余设定：每个responder不共享信息，维护一套自己的clear order，对于自己没标记过clear的room，必须自己去看一圈，才能打上clear的标签，然后执行下一个room的clear工作，自己维护的所有room都clear后返回最近的exit处。因此，r判断下面去哪个room的标准并不是该房间有几个人，在一开始，r不知道room里面有没有人，当其进入了一个room的door，才知道此事room中occupants的layout，并且不会记住，而是下次再来room时，会重新判断，只要没有clear，就迟早会再来。

目前整个log看不出来救人的过程。responder在sweep到occupants时，应该先把o带回到出口，然后再继续清理房间剩下的人，清理完了后再sweep到下一个room，所有room clear完了后

---

- responders都要到一楼的出口才能算结束整个计时过程。
responders在最后回到出口的时候不是回自己来的时候的出口，而是回到当前最近的出口

---

一个room到另一个room的逻辑需要被修改。
1. 路径和距离都要考虑楼层。如果不是同一楼层的，路径和距离都要加上彼此之间的exit作为中转
2. 因此，一个直接的操作就是相邻楼层之间对应位置的exits之间必须reachable以及有频繁的轨迹
3. 不同楼层的room之间不应该有直连的轨迹
4. 刚刚就给你简单说了，你没修好，现在开始深度思考，修改整个寻找下一个room和到达下一个room等逻辑。

---

现在过程中的reachable基本实现了，但是当所有room sweep结束后，responders分别在R4_F2和R5_F2，然后直接回到了E_R_F1,而不是先到E_R_F2再到E_R_F1，这个逻辑需要修改。

---

heatmap.png现在的图画的ok，但是Makespan和order的标签遮挡了地图。将标签移动到图的下方。此外，一楼和二楼之间的距离太大了，二楼看起来像四楼，修改。

---

对于anim.gif，额外保存第一帧，前1/4帧，后1/4帧，最后一帧为png。不要单独绘制，仅仅是抽取这几帧。

---

anim.gif没有问题，但是anim_q3.png和anim_end.png都出现了不从exits，两层楼之间room直接相连的违禁现象，这是为什么？四张图片必须完全来自于gif，再在生成了gif之后抽取出来，而不是重新绘制。

---

1. 现在exit_combo是2^responders种组合，导致计算量非常的大。但是实际上不同的responders是等价的。也就是说在实践中我们只需要把responders平均分成两组，然后决定他们从哪个一楼额exits进入就可以了。因此如果是偶数个responders，那应该有3种组合，如果是奇数个responders，那应该有四种组合。对组进行排列组合即可。

2. 在layout,number of floor,responders.number,相同的组合中，挑选出responders.init_position(从一楼的哪个exits进入)里面time最少的那一条作为结果保存到select_result.csv中。

---


select_result.csv和batch_results.csv的columns顺序修改为layout,floors,per_room_occ,responders,exit_combo,makespan_hms,room_clear_order,exit_combo_id,makespan_s,responder_orders,start_positions

---

1. 所有可视化图例都用英文
2. 一个画布里不要放超过3x3的内容