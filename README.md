# Basketball Backboard
Create a basketball backboard that can make your shot always in

Youtube and Weibo have a very popular video. In simple terms, it is how to design a backboard that can make your shot always ing.
I used Python to implement the code according to the author's ideas.
Judging from the simulation results, this backboard can achieve a perfect shot when shooting from most positions on the court.

Weibo link (with Chinese subtitles): https://weibo.com/2214257545/J0xG8otx5  
Original YouTube link: https://www.youtube.com/watch?v=vtN4tkvcBMA&t=551s

Introduction to the process: First divide the backboard into many small pieces, and then shoot at a parabola with different speeds and different angles at different positions on the basketball court. After hitting the board (hit a small backboard), calculate the pop-up angle and the desired angle respectively, and then adjust this small backboard according to the normal vector. The normal vector that needs to be adjusted for all small backboards and the normal vector before the adjustment form an objective estimation function. Then he kept shooting and adjusted the normal vector of small backboards, and finally got a backboard that could be scored anyway after convergence. But this backboard is still composed of many small blocks, not continuous. In order to be more realistic, these discrete small pieces are fitted into a smooth surface as much as possible.

Technology applied:  
Shot data-> Gaussian distribution sampling (the original author used Monte Carlo)  
Calculate incidence angle and pop-up angle-> High school physical parabolic motion + coordinate system conversion  
Speed after rebound-> High school physical elastic collision  
Error estimation function-> sum of n sub-functions of n small blocks + coordinate ascent algorithm  
Smooth rebound-> Linear regression model  

Problem analysis: In the end, my result is not as beautiful as the original video. It may be because the formula of the parabolic movement is deviated. In addition, I did not consider the motion curve of the first half of the basketball when shooting. On the whole, there are many details that can be improved. For example, the influence of the centrifugal force of the ball rotation on the ejection angle and the influence of the air resistance during the ball movement should also be considered. The algorithm for the physical characteristics is expected.

Youtube和微博有一个很火的视频，简单来说就是如何设计一个无论怎样投篮都能进球的篮板。
我感觉这种基础课程的知识的应用很有意思，就用Python按照作者的思路实现了代码。
从仿真结果来看，这个篮板可以实现从球场大多数位置投篮时的百发百中。

微博链接（有中文字幕）： https://weibo.com/2214257545/J0xG8otx5  
Youtube原始链接：https://www.youtube.com/watch?v=vtN4tkvcBMA&t=551s

流程简介：首先将篮板分成很多小块，然后在篮球场不同位置用不同速度、不同角度以抛物线的方式投篮。打板后（击中一小块篮板）分别计算弹出角度和希望获得的角度，然后按照法向量调整这一小块篮板。将所有小块篮板需要调整的法向量与调整前的法向量组成目标估计函数。然后不停地投篮并调整小块篮板的法向量，最终收敛后获得了无论如何投篮都能进球的篮板。但这个篮板还是由很多小块组成，并不连续。为了更加真实，再将这些离散的小块拟合成一个尽量平滑的曲面。最终结果就是这个百发百中的篮板。

应用到的技术：
投篮数据 –> 高斯分布采样（原作者利用蒙特卡洛）  
计算入射角和弹出角 -> 高中物理抛物线运动 + 坐标系转换  
反弹后速度 -> 高中物理弹性碰撞  
误差估计函数 -> n个小块的n个子函数之和 + 坐标上升算法  
平滑篮板 -> 线性回归模型  

问题分析：最终我的结果没有原始视频那么漂亮，可能因为抛物线运动的公式推导有偏差，另外没有考虑投篮时篮球前半段的运动曲线是不能与篮筐碰撞的。整体来看可以改进的细节还很多，比如还应该考虑球的旋转的离心力对弹出角度的影响，以及球运动时的空气阻力影响，期待物理特性的算法。
