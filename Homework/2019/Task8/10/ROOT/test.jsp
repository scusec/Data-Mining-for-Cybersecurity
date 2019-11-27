<%@ page language="java" import="java.util.*" pageEncoding="UTF-8"%>
<%

Random a = new Random();

out.println((String) session.getAttribute(com.google.code.kaptcha.Constants.KAPTCHA_SESSION_KEY)+"_"+a.nextInt(10000000));
 
%>