#ifndef _debug_included
#define _debug_included

#include <deque>
#include <cmath>
#include <iostream>
#include "utility.h"

#define PRINT(ARR,PREC,WIDTH) std::cout<<std::setprecision(PREC)<<std::setw(WIDTH)<<ARR<<std::endl;

void print_edges(const float_2d &elevations, const std::deque<grid_cell> &low_edges, const std::deque<grid_cell> &high_edges);
void dinf_pit_flows(const float_2d &elevations, float_2d &flowdirs);
void tikz_flowdir_print(const char_2d &flowdirs, std::string filename, float x_scale, float y_scale, float x_offset, float y_offset, bool omit_edges);

template <class T>
static inline T MIN(T a, T b) {
  return (a<b)?a:b;
}

template <class T>
static inline T MAX(T a, T b){
  return (a>b)?a:b;
}

template <class T>
void ddiff(const array2d<T> &arr1, const array2d<T> &arr2, array2d<T> &result){
  diagnostic("Differencing the two arrays...");
  if(arr1.width()!=arr2.width() || arr1.height()!=arr2.height())
    diagnostic("failed! The arrays do not have the same dimensions!\n");
  result.copyprops(arr1);
  //Todo: It isn't guaranteed that subtraction of two elements will avoid the no_data value
  #pragma omp parallel for
  for(int x=0;x<arr1.width();x++)
  for(int y=0;y<arr2.height();y++)
    if(arr1(x,y)==arr1.no_data || arr2(x,y)==arr2.no_data)
      result(x,y)=arr1.no_data;
    else
      result(x,y)=fabs(arr1(x,y)-arr2(x,y));
  diagnostic("success!\n");
}

template <class T>
void dadiff(const array2d<T> &arr1, const array2d<T> &arr2, array2d<T> &result){
  diagnostic("Differencing the two arrays...");
  if(arr1.width()!=arr2.width() || arr1.height()!=arr2.height())
    diagnostic("failed! The arrays do not have the same dimensions!\n");
  result.copyprops(arr1);
  //Todo: It isn't guaranteed that subtraction of two elements will avoid the no_data value
  #pragma omp parallel for
  for(int x=0;x<arr1.width();x++)
  for(int y=0;y<arr2.height();y++)
    if(arr1(x,y)==arr1.no_data || arr2(x,y)==arr2.no_data)
      result(x,y)=arr1.no_data;
    else
      result(x,y)=angdiff_deg(arr1(x,y),arr2(x,y));
  diagnostic("success!\n");
}


template <class T>
void array2d<T>::print_block(std::ostream& out, int minx, int maxx, int miny, int maxy, int precision, std::streamsize swidth){
  out.setf(std::ios::fixed,std::ios::floatfield);
  out<<std::setprecision(precision);

  out<<std::setw(4)<<" "<<"\t";
  for(int x=((minx>0)?minx:0);x<=maxx && x<width();x++)
    out<<std::setw(swidth)<<x<<" ";
  out<<std::endl;

  for(int y=((miny>0)?miny:0);y<=maxy && y<height();y++){
    out<<std::setw(4)<<y<<"\t";
    for(int x=((minx>0)?minx:0);x<=maxx && x<width();x++)
      if(operator()(x,y)==no_data)
        out<<std::setw(swidth)<<"-"<<" ";
      else if(sizeof(T)==1)  //TODO: An ugly way of detecting chars
        out<<std::setw(swidth)<<(int)operator()(x,y)<<" ";
      else
        out<<std::setw(swidth)<<operator()(x,y)<<" ";
    out<<std::endl;
  }
}



template <class T>
void array2d<T>::surroundings(int x0, int y0, int precision) const{
  std::cout.setf(std::ios::fixed,std::ios::floatfield);
  std::cout<<std::setprecision(precision);
  std::cout<<std::endl;

  std::cout<<"Surroundings of ("<<x0<<","<<y0<<")"<<std::endl;
  for(int y=MAX(y0-1,0);y<=MIN(y0+1,height()-1);y++){
    for(int x=MAX(x0-1,0);x<=MIN(x0+1,width()-1);x++)
      std::cout<<std::setw(4)<<operator()(x,y)<<" ";
    std::cout<<std::endl;
  }
  std::cout<<std::endl;
}

//Note: the below cannot detect digital dams based on the metric it uses because the Dinf method may split flows in half so that all cells surrounding one with a high value have less area than that value.
template <class T>    //To be used on flow accumulation DEMs.
int digital_dams(const array2d<T> &dem) {
  int dam_count=0;
  #pragma omp parallel for reduction(+:dam_count)
  for(int x=0;x<dem.width();x++)
  for(int y=0;y<dem.height();y++){
    if(dem(x,y)==dem.no_data) 
      continue;

    bool digital_dam=true;
    for(int n=1;n<=8;n++){
      if(!dem.in_grid(x+dx[n],y+dy[n])) {digital_dam=false;break;}
      if(dem(x+dx[n],y+dy[n])==dem.no_data) {digital_dam=false;break;}
      if(dem(x+dx[n],y+dy[n])>dem(x,y)) {digital_dam=false;break;}
    }
    if(digital_dam){
      dam_count++;
      dem.surroundings(x,y);
    }
  }
  return dam_count;
}

//Note: the below cannot detect digital dams based on the metric it uses because the Dinf method may split flows in half so that all cells surrounding one with a high value have less area than that value.
template <class T>    //To be used on flow accumulation DEMs
int digital_dams_with_angs(const array2d<T> &dem, const array2d<T> &ang) {
  int dam_count=0;
  #pragma omp parallel for reduction(+:dam_count)
  for(int x=0;x<dem.width();x++)
  for(int y=0;y<dem.height();y++){
    if(dem(x,y)==dem.no_data) 
      continue;

    bool digital_dam=true;
    for(int n=1;n<=8;n++){
      if(!dem.in_grid(x+dx[n],y+dy[n])) {digital_dam=false;break;}
      if(dem(x+dx[n],y+dy[n])==dem.no_data) {digital_dam=false;break;}
      if(dem(x+dx[n],y+dy[n])>dem(x,y)) {digital_dam=false;break;}
    }
    if(digital_dam){
      dam_count++;
      diagnostic("Possible digital dam!\n");
      dem.surroundings(x,y);
      ang.surroundings(x,y);
    }
  }
  return dam_count;
}


template <class T>
void print_and_highlight(const array2d<T> &dem, int xh, int yh) {
  std::cerr<<std::endl;
  for(int y=0;y<dem.height();y++){
    for(int x=0;x<dem.width();x++){
      if(xh==x && yh==y)
        std::cerr<<"\033[91m";
      if(sizeof(T)==1)
        std::cerr<<(int)dem(x,y);
      else
        std::cerr<<dem(x,y);
      if(xh==x && yh==y)
        std::cerr<<"\033[39m";
    }
  std::cerr<<std::endl;
  }
}

template <class T>
double avg_diff(const array2d<T> &arr1, const array2d<T> &arr2){
  diagnostic("Finding average difference between two arrays...");
  if(arr1.width()!=arr2.width() || arr1.height()!=arr2.height())
    diagnostic("failed! The arrays do not have the same dimensions!\n");

  double diff=0;
  int ccount=0;
  double maxdiff=0;
  #pragma omp parallel for collapse(2) reduction(+:diff) reduction(+:ccount) reduction(max:maxdiff)
  for(int x=0;x<arr1.width();x++)
  for(int y=0;y<arr2.height();y++){
    double temp_diff;
    if(arr1(x,y)==arr1.no_data || arr2(x,y)==arr2.no_data)
      continue;
    if(arr1(x,y)!=arr1(x,y) || arr2(x,y)!=arr2(x,y))  //If it's NaN
      continue;
    temp_diff=fabs(arr1(x,y)-arr2(x,y));

    diff+=temp_diff;
    ccount++;

    if(temp_diff>maxdiff)
      maxdiff=temp_diff;
  }
  diagnostic("success!\n");
  diagnostic_arg("Maximal difference was: %lf\n",maxdiff);

  return (float)(diff/ccount);
}

#endif