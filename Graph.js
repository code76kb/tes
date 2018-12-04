
import React,{Component} from 'react';
import {StyleSheet,ART,View} from 'react-native';
import {translate} from '../config/localization.js';
import font from '../styles/font.js';
import colors from '../styles/colors';
import AllStyles from '../styles/AllStyles.js';

const {Surface,Shape,Group,Text,Path} = ART;

const TAG = "Graph :";
const tan30 = 0.577350269;
const cost30 = 0.866025404;
const tan60 = 1.732050808;
export default class Graph extends Component {

  constructor(props){
     super(props)
  //outer Triangle
    this.point1={x:0,y:0};
    this.point2={x:0,y:0};
    this.point3={x:0,y:0};
    this.point4={x:0,y:0};
 // Skill Text
    this.text_x_offset=60;
    this.text_y_offset=30
    this.txt_technical_cord={x:0,y:0};
    this.txt_mental_cord={x:0,y:0};
    this.txt_physical_cord={x:0,y:0};

//To-Do Txt
    this.txt_to_do_cord={x:0,y:0};

 // Inner Traingels
  this.innerTraingelsCords=[  {p1:{x:0,y:0},p2:{x:0,y:0},p3:{x:0,y:0}},
                              {p1:{x:0,y:0},p2:{x:0,y:0},p3:{x:0,y:0}},
                              {p1:{x:0,y:0},p2:{x:0,y:0},p3:{x:0,y:0}},
                              {p1:{x:0,y:0},p2:{x:0,y:0},p3:{x:0,y:0}},
                              {p1:{x:0,y:0},p2:{x:0,y:0},p3:{x:0,y:0}},
                              {p1:{x:0,y:0},p2:{x:0,y:0},p3:{x:0,y:0}},
                              {p1:{x:0,y:0},p2:{x:0,y:0},p3:{x:0,y:0}},
                              {p1:{x:0,y:0},p2:{x:0,y:0},p3:{x:0,y:0}},
                              {p1:{x:0,y:0},p2:{x:0,y:0},p3:{x:0,y:0}},
                              {p1:{x:0,y:0},p2:{x:0,y:0},p3:{x:0,y:0}},
                           ];
  this.inner_traingels_offset=30;

  //Skill Graph coords
  this.technical_Skill_Coord = {x:0,y:0};
  this.mental_Skill_Coord    = {x:0,y:0};
  this.physical_Skill_Coord  = {x:0,y:0};

  this.txt_technical_Skill_Coord = {x:0,y:0};
  this.txt_mental_Skill_Coord    = {x:0,y:0};
  this.txt_physical_Skill_Coord  = {x:0,y:0};

  this.Tech = props.data.technical;      // {/* on P1 axis*/}
  this.Mental = props.data.mental;    // {/* on P3 axis*/}
  this.Physical = props.data.physical;  // {/* on P2 axis*/}

  console.log(TAG,"Constructor Graph data :"+this.Tech+' '+this.Mental+' '+this.Physical);

  }

  init(){
    console.log(TAG+'init call');
    this.point1={x:0,y:0};
    this.point2={x:0,y:0};
    this.point3={x:0,y:0};
    this.point4={x:0,y:0};
  }

componentWillReceiveProps(nextprops){
  // console.log(TAG+'props will recieve');

  this.Tech = nextprops.data.technical;      // {/* on P1 axis*/}
  this.Mental = nextprops.data.mental;    // {/* on P3 axis*/}
  this.Physical = nextprops.data.physical;  // {/* on P2 axis*/}

  // console.log(TAG,"On recieve Props Graph data :"+this.Tech+' '+this.Mental+' '+this.Physical);
  this.doMath();

  // if (this.props.width != nextprops.width && this.props.height != nextprops.height) {
  //   // this.props = nextprops;
  //   // this.doMath();
  // }
  // else {
  //   console.log(TAG+'No re-rander');
  // }
}

// shouldComponentUpdate(nextProps) {
//   console.log(TAG+'should component update.');
//   if (this.props.width != nextProps.width && this.props.height != nextProps.height) {
//     this.props = nextProps;
//     this.doMath();
//     return true;
//   }
//   else {
//     console.log(TAG+'No re-rander');
//     return false;
//     }
//    }


doMath = ()=>{
  this.init();
  //window dimention
  var width = this.props.width; // X axis
  var height = this.props.height; // Y axis
  console.log(TAG+'W: '+width," H: "+height);
  if (width === 0 || height === 0)
   return;

  //start point will be "one third of the width" on X-axis and "0" on Y-axis

// calculate OuterTraingel cords
  var x_genesis = width/3
  var y_cutOff =  Math.sqrt(Math.pow(width - x_genesis,2) - Math.pow((width-x_genesis)/2 , 2)); //height-(height/6);
  var L = tan30*y_cutOff;

 console.log(TAG+"variables:"+" x_genesis: "+x_genesis+" y_cutoff: "+y_cutOff+" L: "+L);

  this.point1.x = x_genesis
  this.point1.y = 0;

  this.point2.x = L+x_genesis;
  this.point2.y = y_cutOff;

  this.point3.x = x_genesis + Math.sqrt(Math.pow(this.point1.x - this.point2.x,2) + Math.pow(this.point1.y - this.point2.y,2));
  this.point3.y = 0

  this.point4.x = this.point2.x-(tan30*(height-y_cutOff))
  this.point4.y = height;


  var side1 = Math.sqrt(Math.pow((this.point1.x - this.point3.x),2) + Math.pow((this.point1.y - this.point3.y),2) ).toFixed(4);
  var side2 = Math.sqrt(Math.pow((this.point2.x - this.point1.x),2) + Math.pow((this.point2.y - this.point1.y),2) ).toFixed(4);
  var side3 = Math.sqrt(Math.pow((this.point2.x - this.point3.x),2) +  Math.pow((this.point2.y - this.point3.y),2) ).toFixed(4);
  var isOuterTaingleValid = false;
  if (side1==side2 && side3==side1)
    isOuterTaingleValid =true;
  // console.log(TAG+"Outer Traingel is Valid :"+isOuterTaingleValid);
  // console.log(TAG+"Side 1:"+side1+" Side 2: "+side2+" side 3:"+side3);
  //
  // console.log(TAG+"Point 1 :",JSON.stringify(this.point1));
  // console.log(TAG+"Point 2 :",JSON.stringify(this.point2));
  // console.log(TAG+"Point 3 :",JSON.stringify(this.point3));
  // console.log(TAG+"Point 4 :",JSON.stringify(this.point4));


  //new Math
var inner_padding = 20;

var  x0 = this.point2.x - this.point1.x;
  var y_plan_for_p1_p3 = x0 * tan30;
  var y_plan_for_p2 = this.point2.y - y_plan_for_p1_p3;

  var x_plan_for_p1 = this.point2.x - this.point1.x;
  var x_plan_for_p3 = x_plan_for_p1;

  // console.log(TAG,"y-Plan :"+y_plan_for_p2+" :"+y_plan_for_p1_p3);
  // console.log(TAG,"x-Plan :"+x_plan_for_p1+" :"+x_plan_for_p3);

  var y_step_size_p1_p3 = (y_plan_for_p1_p3 - inner_padding) / 10;
  var y_step_size_p2    = (y_plan_for_p2 - inner_padding) / 10;
  var x_step_size_p1    = (x_plan_for_p1 - inner_padding) /10;
  var x_step_size_p3    = (x_plan_for_p3 - inner_padding) /10;

var padd = 3;
  for (var i=0; i <10; i++){

  this.innerTraingelsCords[i].p1.x = this.point1.x + (x_step_size_p1 * i) + (padd * x_step_size_p1);
  this.innerTraingelsCords[i].p1.y = this.point1.y + (y_step_size_p1_p3 * i) + (padd * y_step_size_p1_p3);

  this.innerTraingelsCords[i].p2.x = this.point2.x;
  this.innerTraingelsCords[i].p2.y = this.point2.y - (y_step_size_p2 * i) - (padd * y_step_size_p2);

  this.innerTraingelsCords[i].p3.x = this.point3.x - (x_step_size_p3 * i) - (padd * x_step_size_p3);
  this.innerTraingelsCords[i].p3.y = this.point3.y + (y_step_size_p1_p3 * i) + (padd * y_step_size_p1_p3);

}

//Text coords
 this.txt_technical_cord.x = this.innerTraingelsCords[0].p1.x;
 this.txt_technical_cord.y = this.innerTraingelsCords[0].p1.y - 12;

 this.txt_mental_cord.x    = this.innerTraingelsCords[0].p3.x;
 this.txt_mental_cord.y = this.innerTraingelsCords[0].p3.y - 12;

 this.txt_physical_cord.x = this.innerTraingelsCords[0].p2.x;
 this.txt_physical_cord.y = this.innerTraingelsCords[0].p2.y;

 //to-do Cord
 this.txt_to_do_cord.x = this.point2.x + 12;
 this.txt_to_do_cord.y = this.point2.y;

 //Compute graph by given score
 // for test skillPoints
this.technical_Skill_Coord.x = this.innerTraingelsCords[ 9 - (this.Tech)].p1.x;
this.technical_Skill_Coord.y = this.innerTraingelsCords[ 9 - (this.Tech)].p1.y;

this.mental_Skill_Coord.x = this.innerTraingelsCords[ 9 - (this.Mental)].p3.x;
this.mental_Skill_Coord.y = this.innerTraingelsCords[ 9 - (this.Mental)].p3.y;

this.physical_Skill_Coord.x = this.innerTraingelsCords[ 9 - (this.Physical)].p2.x;
this.physical_Skill_Coord.y = this.innerTraingelsCords[ 9 - (this.Physical)].p2.y;

//Skill points
this.txt_technical_Skill_Coord.x = this.technical_Skill_Coord.x - 10;
this.txt_technical_Skill_Coord.y = this.technical_Skill_Coord.y + 4;

this.txt_physical_Skill_Coord.x = this.physical_Skill_Coord.x;
this.txt_physical_Skill_Coord.y = this.physical_Skill_Coord.y + 4;

this.txt_mental_Skill_Coord.x = this.mental_Skill_Coord.x + 6;
this.txt_mental_Skill_Coord.y = this.mental_Skill_Coord.y + 4;

this.setState({triger:''});

}

getPath=()=>{
  return(
     new Path()
     .moveTo(this.point4.x,this.point4.y)
     .lineTo(this.point3.x,this.point3.y)
     .lineTo(this.point1.x,this.point1.y)
     .lineTo(this.point2.x,this.point2.y)
     .close()
  );
}

getPathInner=(i)=>{
  return(
     new Path()
     .moveTo(this.innerTraingelsCords[i].p1.x , this.innerTraingelsCords[i].p1.y)
     .lineTo(this.innerTraingelsCords[i].p2.x , this.innerTraingelsCords[i].p2.y)
     .lineTo(this.innerTraingelsCords[i].p3.x , this.innerTraingelsCords[i].p3.y)
     .close()
  );
}

getSkillPath () {
  return(
    new Path()
    .moveTo(this.technical_Skill_Coord.x , this.technical_Skill_Coord.y)
    .lineTo(this.physical_Skill_Coord.x , this.physical_Skill_Coord.y)
    .lineTo(this.mental_Skill_Coord.x , this.mental_Skill_Coord.y)
    .close()
  );
}

handleResponderGrant(evt){
  event = evt.nativeEvent;
  //P4 to P3
  //lal
  //kka
  var l  = event.locationY;
  var a  = l / tan60;
  var x1 = this.point3.x - a;
  var x0 = this.point1.x + a;

console.log('you are touching at :'+event.locationX+","+event.locationY+' x1=>'+x1);

  if (x1 < event.locationX){
      console.log(TAG+'touch on To-Do');
      this.props.callback('todo');
    }
  else if (x0 < event.locationX ) {
    console.log(TAG+'touch on Skill !');
    this.props.callback('skill');
  }

}

//Re-rander Graph
re_rander(){
  this.setState({triger:''});
  console.log(TAG+'Re-Rander');
}

 render(){
   // console.log(TAG+'render called !!!!!');

   return(

   <View
       onStartShouldSetResponderCapture={(eve)=>{return true}}
       onMoveShouldSetResponderCapture={(eve)=>{return true}}
       onStartShouldSetResponder={(eve)=>{return true}}
       onMoveShouldSetResponder ={(eve)=>{return true}}
      onResponderGrant={this.handleResponderGrant.bind(this)} >
      <Surface width={this.props.width} height={this.props.height} style={styles.surface}>

          <Group>
               {/*Big dark traingle  */}
                <Shape
                       d={this.getPath()}
                       stroke={colors.blackc}
                       strokeWidth={4}
                       />

                {/*Skill tags text */}
                <Text font={{fontSize:10,fontFamily:font.app_font,fontWeight:'bold'}}
                      fill={colors.blackc}
                      x={this.txt_technical_cord.x}
                      y={this.txt_technical_cord.y}
                      alignment={'left'}>TECHNICAL</Text>

                <Text font={{fontSize:10,fontFamily:font.app_font,fontWeight:'bold'}}
                      fill={colors.blackc}
                      x={this.txt_mental_cord.x}
                      y={this.txt_mental_cord.y}
                      alignment={'right'}>MENTAL</Text>

                <Text font={{fontSize:10,fontFamily:font.app_font,fontWeight:'bold'}}
                      fill={colors.blackc}
                      x={this.txt_physical_cord.x}
                      y={this.txt_physical_cord.y}
                      alignment={'right'}
                      transform={new ART.Transform().rotateTo(60)}>PHYSICAL</Text>

              {/*To-Do  */}
              <Text font={{fontSize:35,fontFamily:font.app_font,fontWeight:'bold'}}
                    fill={colors.blackc}
                    x={this.txt_to_do_cord.x}
                    y={this.txt_to_do_cord.y}
                    alignment={'left'}
                    transform={new ART.Transform().rotateTo(-60)}>To-Do</Text>

              {/* Skill Text */}
              <Text font={{fontSize:10,fontFamily:font.app_font,fontWeight:'bold'}}
                    fill={colors.blackc}
                    x={this.txt_technical_Skill_Coord.x}
                    y={this.txt_technical_Skill_Coord.y}
                    alignment={'left'}>{(this.Tech+1).toString()}</Text>

              <Text font={{fontSize:10,fontFamily:font.app_font,fontWeight:'bold'}}
                    fill={colors.blackc}
                    x={this.txt_physical_Skill_Coord.x}
                    y={this.txt_physical_Skill_Coord.y}
                    alignment={'center'}>{(this.Physical+1).toString()}</Text>

              <Text font={{fontSize:10,fontFamily:font.app_font,fontWeight:'bold'}}
                    fill={colors.blackc}
                    x={this.txt_mental_Skill_Coord.x}
                    y={this.txt_mental_Skill_Coord.y}
                    alignment={'right'}>{(this.Mental+1).toString()}</Text>


                {/* Inner Trainangels */}
                <Shape
                       d={this.getPathInner(0)}
                       stroke={colors.blackc}
                       strokeWidth={2}
                       />
                       <Shape
                              d={this.getPathInner(1)}
                              stroke={colors.blackc}
                              strokeWidth={1}
                              strokeDash={[0,5]}
                              strokeCap={"square"}
                              strokeJoin={"bevel"}
                              />
                              <Shape
                                     d={this.getPathInner(2)}
                                     stroke={colors.blackc}
                                     strokeWidth={1}
                                     strokeDash={[0,5]}
                                     strokeCap={"square"}
                                     strokeJoin={"bevel"}
                                     />

                                     <Shape
                                            d={this.getPathInner(3)}
                                            stroke={colors.blackc}
                                            strokeWidth={1}
                                            strokeDash={[0,5]}
                                            strokeCap={"square"}
                                            strokeJoin={"bevel"}
                                            />
                                            <Shape
                                                   d={this.getPathInner(4)}
                                                   stroke={colors.blackc}
                                                   strokeWidth={1}
                                                   strokeDash={[0,5]}
                                                   strokeCap={"square"}
                                                   strokeJoin={"bevel"}
                                                   />
                                                   <Shape
                                                          d={this.getPathInner(5)}
                                                          stroke={colors.blackc}
                                                          strokeWidth={1}
                                                          strokeDash={[0,5]}
                                                          strokeCap={"square"}
                                                          strokeJoin={"bevel"}
                                                          />
                                                          <Shape
                                                                 d={this.getPathInner(6)}
                                                                 stroke={colors.blackc}
                                                                 strokeWidth={1}
                                                                 strokeDash={[0,5]}
                                                                 strokeCap={"square"}
                                                                 strokeJoin={"bevel"}
                                                                 />
                                                                 <Shape
                                                                        d={this.getPathInner(7)}
                                                                        stroke={colors.blackc}
                                                                        strokeWidth={1}
                                                                        strokeDash={[0,5]}
                                                                        strokeCap={"square"}
                                                                        strokeJoin={"bevel"}
                                                                        />
                                                                        <Shape
                                                                               d={this.getPathInner(8)}
                                                                               stroke={colors.blackc}
                                                                               strokeWidth={1}
                                                                               strokeDash={[0,5]}
                                                                               strokeCap={"square"}
                                                                               strokeJoin={"bevel"}
                                                                               />
                                                                               <Shape
                                                                                      d={this.getPathInner(9)}
                                                                                      stroke={colors.blackc}
                                                                                      strokeWidth={1}
                                                                                      strokeDash={[0,5]}
                                                                                      strokeCap={"square"}
                                                                                      strokeJoin={"bevel"}
                                                                                      />


          {/* Skill plot */}
          <Shape
            d={this.getSkillPath()}
            stroke={colors.orangec}
            strokeWidth={1}
            fill={colors.blackc}
           />

          </Group>

      </Surface>
  </View>
   );
 }

}

const styles = StyleSheet.create({
  surface:{
    backgroundColor:'transparent',
  }
});
