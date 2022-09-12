---
layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <title>Learning to Walk by Steering: Perceptive Quadrupedal Locomotion in Dynamic Environments</title>


<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->
<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>

<!-- Global site tag (gtag.js) - Google Analytics -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Titillium Web","HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }
  
  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
  }
  
IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table 
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


</head>

<body data-gr-c-s-loaded="true">


<style>
a {
  color: #bf5700;
  text-decoration: none;
}
</style>


<style>
highlight {
  color: #ff0000;
  text-decoration: none;
}
</style>

<div id="primarycontent">
<center><h1><strong>Learning to Walk by Steering: Perceptive Quadrupedal Locomotion in Dynamic Environments</strong></h1></center>
<center><h2><strong>
    <a href="https://mingyoseo.com/">Mingyo Seo</a>&nbsp;&nbsp;&nbsp;
    <a>Ryan Gupta</a>&nbsp;&nbsp;&nbsp;
    <a href="https://zhuyifengzju.github.io/">Yifeng Zhu</a>&nbsp;&nbsp;&nbsp;
    <a href="https://alexyskoutnev.github.io/alexyskoutnev-github.io/">Alexy Skoutnev</a>&nbsp;&nbsp;&nbsp;<br>
    <a href="https://www.ae.utexas.edu/people/faculty/faculty-directory/sentis">Luis Sentis</a>&nbsp;&nbsp;&nbsp; 
    <a href="https://cs.utexas.edu/~yukez">Yuke Zhu</a>&nbsp;&nbsp;&nbsp;
   </strong></h2>
    <center><h2><strong>
        <a href="https://www.cs.utexas.edu/">The University of Texas at Austin</a>&nbsp;&nbsp;&nbsp;   
    </strong></h2></center>

	<center><h2><strong><a href="">Paper</a> | <a href="https://github.com/UT-Austin-RPL/PRELUDE">Code</a> | <a href="./src/bib.txt">Bibtex</a> </strong></h2></center>

 <center><p><span style="font-size:20px;"></span></p></center>
<!-- <p> -->
<!--   </p><table border="0" cellspacing="10" cellpadding="0" align="center">  -->
<!--   <tbody> -->
<!--   <tr> -->
<!--   <\!-- For autoplay -\-> -->
<!-- <iframe width="560" height="315" -->
<!--   src="https://www.youtube.com/embed/GCfs3DJ4aO4?autoplay=1&mute=1&loop=1" -->
<!--   autoplay="true" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>   -->
<!--   <\!-- No autoplay -\-> -->
<!-- <\!-- <iframe width="560" height="315" -\-> -->
<!-- <\!--   src="https://www.youtube.com/embed/GCfs3DJ4aO4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>   -\-> -->

<!-- </tr> -->
<!-- </tbody> -->
<!-- </table> -->

<table border="0" cellspacing="10" cellpadding="0" align="center"> 
  <tbody>
    <tr> 
      <td align="center" valign="middle">
        <iframe width="800" height="450" src="https://www.youtube.com/embed/rdiDvBMQSrg?showinfo=0&playlist=rdiDvBMQSrg&autoplay=1&loop=1" autoplay="true" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </td>
    </tr>
  </tbody>
</table>

<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
We tackle the problem of perceptive locomotion in dynamic environments. In this problem, a quadrupedal robot must exhibit robust and agile walking behaviors in response to environmental clutter and moving obstacles. We present a hierarchical learning framework, named <b>PRELUDE</b>, which decomposes the problem of perceptive locomotion into high-level decision-making to predict navigation commands and low-level gait generation to realize the target commands. In this framework, we train the high-level navigation controller with imitation learning on human demonstrations collected on a steerable cart and the low-level gait controller with reinforcement learning (RL). Therefore, our method can acquire complex navigation behaviors from human supervision and discover versatile gaits from trial and error. We demonstrate the effectiveness of our approach in simulation and with hardware experiments.
</p></td></tr></table>
</p>
  </div>
</p>

<hr>

<h1 align="center">Method Overview</h1>

  <table border="0" cellspacing="10" cellpadding="0" align="center"> 
    <tbody>
      <tr> 
        <td align="center" valign="middle">
          <a href="./src/approach.gif"><img src="./src/approach.gif" style="width:100%;"> </a>
        </td>
      </tr>
    </tbody>
  </table>

  <table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  Overview of PRELUDE. We introduce a control hierarchy where the high-level controller, trained with imitation learning, sets navigation commands and the low-level gait controller, trained with reinforcement learning, realizes the target commands through joint-space actuation. This combination enables us to effectively deploy the entire hierarchy on quadrupedal robots in real-world environments.
</p></td></tr></table>

  
<br><br><hr> <h1 align="center">Hierarchical Perceptive Locomotion Model</h1> <!-- <h2
align="center"></h2> --> <table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle"><a href="./src/pipeline.png"> <img
src="./src/pipeline.png" style="width:100%;"> </a></td>
</tr> </tbody> </table>

<table width=800px><tr><td> <p align="justify" width="20%">The high-level navigation policy generates the target velocity command at 10Hz from the onboard RGB-D camera observation and robot heading. The target velocity command, including linear and angular velocities, is used as input to the low-level gait controller along with the buffer of recent robot states. The low-level gait policy predicts the joint-space actions as the desired joint positions at 38Hz and sends them to the quadruped robot for actuation.
</p></td></tr></table>
<br>

<hr>

<h1 align="center">Real Robot Evaluation</h1>

  <table width=800px><tr><td> <p align="justify" width="20%">We perform real-world trials where the robot traverses 15m-length tracks in different configurations. We compare it with our self-baseline PRELUDE (A1 Default Gait), a variant of our final model, using the robot’s default model-based controller instead. 
  PRELUDE (Ours) tracks trajectories more robustly (with a <highlight>1.11x</highlight> longer average length traversed and a <highlight>20%</highlight> increase in success rate) than PRELUDE (A1 Default Gait).
  We observed that PRELUDE (A1 Default Gait) drifts aggressively after a high-speed turning and collides into the wall, while PRELUDE (Ours) turns rapidly to bypass the walking crowd and completes the trial successfully.

  </p></td></tr></table>

  <table border="0" cellspacing="10" cellpadding="0" align="center"> 
    <tbody>
      <tr> 
        <td align="center" valign="middle">
          <iframe width="798" height="300" src="https://www.youtube.com/embed/csr5hi5v_Bs?autoplay=1&mute=1&playlist=csr5hi5v_Bs&loop=1" autoplay="true" frameborder="5" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </td>
      </tr>
    </tbody>
  </table>

<hr>

<h1 align="center">Deploying in Unseen Environments</h1>

  <table width=800px><tr><td> <p align="justify" width="20%">
  We deployed in unseen human-centered environments with static and dynamic obstacles. It exhibits robust locomotion behaviors with on-board visual perception.
  </p></td></tr></table>

  <table border="0" cellspacing="10" cellpadding="0" align="center"> 
    <tr>
        <td align="center" valign="middle">
          <!-- <iframe width="450" height="253" src="https://www.youtube.com/embed/K3pYobHhzDs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> -->
          <iframe width="600" height="337" src="https://www.youtube.com/embed/K3pYobHhzDs?autoplay=1&mute=1&playlist=K3pYobHhzDs&loop=1" autoplay="true" frameborder="5" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </td>
      </tr>
          <!-- <iframe width="450" height="253" src="https://www.youtube.com/embed/UfZjapJBbUs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> -->
          <!-- <iframe width="450" height="253" src="https://www.youtube.com/embed/UfZjapJBbUs?autoplay=1&mute=1&playlist=UfZjapJBbUs&loop=1" autoplay="true" frameborder="5" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> -->
      <tr>
        <td align="center" valign="middle">
          <!-- <iframe width="450" height="253" src="https://www.youtube.com/embed/cc3b8VM7Jb0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> -->
          <iframe width="600" height="337" src="https://www.youtube.com/embed/cc3b8VM7Jb0?autoplay=1&mute=1&playlist=cc3b8VM7Jb0&loop=1" autoplay="true" frameborder="5" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </td>
      </tr>
      <tr>
        <td align="center" valign="middle">
          <!-- <iframe width="450" height="253" src="https://www.youtube.com/embed/9yEtgGHy9Aw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> -->
          <iframe width="600" height="337" src="https://www.youtube.com/embed/9yEtgGHy9Aw?autoplay=1&mute=1&playlist=9yEtgGHy9Aw&loop=1" autoplay="true" frameborder="5" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
        </td>
      </tr>
  </table>




<br><hr> <table align=center width=800px> <tr> <td> <left>
<center><h1>Acknowledgements</h1></center> -
<!-- The webpage template was borrowed from some <a href="https://nvlabs.github.io/SPADE/">GAN folks</a>. -->
</left></td></tr></table>
<br><br>

<div style="display:none">
<!-- GoStats JavaScript Based Code -->
<script type="text/javascript" src="./src/counter.js"></script>
<script type="text/javascript">_gos='c3.gostats.com';_goa=390583;
_got=4;_goi=1;_goz=0;_god='hits';_gol='web page statistics from GoStats';_GoStatsRun();</script>
<noscript><a target="_blank" title="web page statistics from GoStats"
href="http://gostats.com"><img alt="web page statistics from GoStats"
src="http://c3.gostats.com/bin/count/a_390583/t_4/i_1/z_0/show_hits/counter.png"
style="border-width:0" /></a></noscript>
</div>
<!-- End GoStats JavaScript Based Code -->
<!-- </center></div></body></div> -->

