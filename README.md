# GraphEvac
GraphEvac by NFX

本项目给出一个基于 ILP 的“建筑清扫（sweep）/救援调度”最优化模型，并提供经验参数（速度、通信成功率、负重能力等）来更贴近真实应急环境。下面整理关键设定与对应的数学公式，便于复现实验与调整参数。

## 记号与集合
- 房间与节点：房间集合记为 $\mathcal{R}$，响应者集合为 $\mathcal{K}$，出口节点集合为 $\mathcal{E}$。
- 坐标与距离：每个节点 $i$ 有坐标 $\mathrm{coord}(i)$。
  - 若为二维坐标 $\mathrm{coord}(i)=(x_i,z_i)$，则两点欧氏距离为
    $$d_{ij}=\sqrt{(x_i-x_j)^2+(z_i-z_j)^2}\,.$$
  - 若为一维坐标 $x_i$，则 $d_{ij}=|x_i-x_j|$。
- 人数：房间 $r$ 的被救人数为 $N_r$。

## 速度与能力（默认值/范围）
- 占用人（被救者）速度：
  - 高能见度：$v_{\text{occ,high}}=0.5\,\mathrm{m/s}$（普通人），健壮者可取 $0.6\,\mathrm{m/s}$。
  - 低能见度：$v_{\text{occ,low}}\in[0.2,0.4]\,\mathrm{m/s}$（示例默认取 $0.3$）。
- 响应者（消防员）速度：
  - 搜索/任务中：$v_{\text{search}}\in[0.4,0.6]\,\mathrm{m/s}$（示例默认取 $0.5$）。
  - 轻装、清晰通道：$v_{\max} \approx 1.0\,\mathrm{m/s}$（非搜索态，可用于其他场景）。
  - 搬运/拖带：$v_{\text{carry}}\in[0.2,0.3]\,\mathrm{m/s}$（示例默认取 $0.25$）。
- 搬运能力（分组）：$C\in\{2,3,4\}$（示例默认 $C=3$）。
- 通信/引导成功率：$p_{\text{comm}}\approx 0.85$。
- 基础检查时间：每房间固定时间 $t_{\text{base}}$（示例默认 $5\,\mathrm{s}$）。

代码中的默认配置可在 `src/ilp_sweep.py` 主程序中直接修改：
- `occupant_speed_high=0.5`，`occupant_speed_low=0.3`
- `responder_speed_search=0.5`，`responder_speed_carry=0.25`
- `carry_capacity=3`，`comm_success=0.85`
- `empirical_mode` 取值：`None`、`"guide_high"`、`"guide_low"`、`"carry"`

## 行走时间（节点间）
响应者在图上从节点 $i$ 到 $j$ 的行走时间为
$$t_{ij}=\frac{d_{ij}}{v_{\text{walk}}}\,.$$

当启用经验模式（`empirical_mode\neq\text{None}`）时，路径行走速度默认采用搜索速度 $v_{\text{walk}}=v_{\text{search}}$；否则使用输入的 `walk_speed`。

## 房间清扫时间模型
记房间 $r$ 到任一出口的最近距离为
$$d_r = \min_{e\in\mathcal{E}} d(r,e)\,.$$

提供两类时间模型：线性基线模型与经验模型。

1) 线性基线模型（不启用经验模式）
$$t_r\,=\,t_{\text{base}}\,+\,\alpha\,N_r\,,$$
其中 $\alpha$ 为“每人额外时间”（代码中的 `time_per_occupant`）。

2) 经验模型（启用 `empirical_mode`）
- 引导（高能见度）：
  $$v_{\text{pair}}=\min\bigl(v_{\text{occ,high}},\,v_{\text{search}}\bigr)\,,$$
  $$t_r\,=\,t_{\text{base}}\,+\,\frac{N_r\,d_r}{v_{\text{pair}}\,p_{\text{comm}}}\,.$$
- 引导（低能见度）：
  $$v_{\text{pair}}=\min\bigl(v_{\text{occ,low}},\,v_{\text{search}}\bigr)\,,$$
  $$t_r\,=\,t_{\text{base}}\,+\,\frac{N_r\,d_r}{v_{\text{pair}}\,p_{\text{comm}}}\,.$$
- 搬运/拖带（分组摆渡）：
  $$G_r=\left\lceil\frac{N_r}{C}\right\rceil\,,\quad t_{\text{shuttle}}=\frac{2\,d_r}{v_{\text{carry}}}\,,$$
  $$t_r\,=\,t_{\text{base}}\,+\,\frac{G_r\,t_{\text{shuttle}}}{p_{\text{comm}}}\,.$$

注：引导模式按“人-员速度取较小值”估算同行速度；搬运模式按“往返摆渡+分组”估算总时间，上式均以 $p_{\text{comm}}$ 作时间膨胀（通信/组织的额外耗时）。

## 目标与完工时间（汇总）
- 单个响应者 $k$ 的总时间下界：
  $$T_k\;\ge\;\sum_{(i,j)} t_{ij}\,y_{kij}\;+\;\sum_{r\in\mathcal{R}} t_r\,x_{rk}\,,$$
  其中 $y_{kij}\in\{0,1\}$ 表示 $k$ 是否行走弧 $(i\to j)$，$x_{rk}\in\{0,1\}$ 表示 $r$ 是否指派给 $k$。
- 项目最小化“工期”（完工时间/最大时间）：
  $$\min\;T_{\text{total}}\quad\text{s.t.}\quad T_{\text{total}}\;\ge\;T_k\;\;\forall k\in\mathcal{K}\,,$$
  即 $$T_{\text{total}} = \max_{k\in\mathcal{K}} T_k\,.$$

## 冗余清扫模式（不共享信息）
设每个响应者 $k$ 独立清扫全部房间：从其起点出口 $s_k$ 出发，访问 $\mathcal{R}$ 中每个房间恰好一次，最终返回最近的出口。为便于建模，引入终点汇点 $g_k$（仅作“返回最近出口”的时间计费，不可再离开）。

- 节点与弧：$\{s_k\}\cup\mathcal{R}\cup\{g_k\}$。
- 行走时间：
  - $t_{s_k,r}=\dfrac{d(s_k,r)}{v_{\text{walk}}}$；
  - $t_{ij}=\dfrac{d_{ij}}{v_{\text{walk}}}$，$i\ne j\in\mathcal{R}$；
  - $t_{r,g_k}=\min\limits_{e\in\mathcal{E}}\dfrac{d(r,e)}{v_{\text{walk}}}$（返回最近出口）。
- 流量与度约束：
  - 起点出度：$$\sum_{r\in\mathcal{R}} y_{k,s_k r}=1\,.$$ 
  - 终点入度：$$\sum_{r\in\mathcal{R}} y_{k,r g_k}=1\,.$$ 
  - 每房间入/出各一次：$$\sum_{i\in\mathcal{R}\setminus\{r\}} y_{k,ir}+y_{k,s_k r}=1\,,\quad \sum_{j\in\mathcal{R}\setminus\{r\}} y_{k,rj}+y_{k,r g_k}=1\,.$$
  - 子回路消除：对 $i\ne j\in\mathcal{R}$ 用 MTZ 约束 $$u_{k,i}-u_{k,j}+M\,y_{k,ij}\le M-1\,,\quad 1\le u_{k,r}\le |\mathcal{R}|\,.$$
- 单个响应者时间下界：
  $$T_k\;\ge\;\sum_{(i,j)} t_{ij}\,y_{kij}\;+\;\sum_{r\in\mathcal{R}} t_r\,,$$
  其中 $\sum_{r} t_r$ 表示“$k$ 自己也要将所有房间清扫一遍”（不共享信息的冗余代价）。
- 工期：同上 $$T_{\text{total}}=\max_k T_k\,.$$

在代码中，通过 `redundancy_mode="per_responder_all_rooms"` 启用该模式；此时即使不同房间清扫时间 $t_r$ 互不相同，它们对路径选择只提供“常数”代价（每位响应者都要清扫所有房间），因此路径主要由图上行走距离决定，符合“事先未知房间人数，不以人数决定下一间去哪”的设定。若需进一步忽略人数对 $t_r$ 的影响，可将 `empirical_mode=None`，仅保留 $t_{\text{base}}$。

## 配置与运行
- 配置文件：`src/configs.py`
  - 场景与参数集中管理：房间、坐标、人数、模式与速度等。
  - 关键键值：
    - `rooms, responders, start_node, coords, occupants`
    - `empirical_mode` ∈ {`None`, `"guide_high"`, `"guide_low"`, `"carry"`}
    - `redundancy_mode` ∈ {`"assignment"`, `"per_responder_all_rooms"`}
    - `base_check_time, time_per_occupant, walk_speed`
    - `occupant_speed_high/low, responder_speed_search/carry, carry_capacity, comm_success`
- 安装与运行：
  - `python3 -m pip install -r requirements.txt`
  - `python3 src/ilp_sweep.py`
  - 产物：控制台结果 + `out/topo_paths.png`, `out/gantt.png`, `out/anim.gif`

## 说明
- 上述火灾/气体蔓延速度未纳入当前 ILP，但可在未来以时间步长对危险区域加约束或降低当地速度来扩展模型。


```
brew install xyz
python src/batch.py \
        --floors 1-18 \
        --layouts T,L \
        --occ 1-10 \
        --resp 1-10 \
        --out output/batch_results.csv

FLOORS=2 make run
```