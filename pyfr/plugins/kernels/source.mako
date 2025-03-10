<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='source' params='t, u, ploc, src' externs='omega_dot, neg_omg, rotating_U_dot, rotating_V_dot'>

src[0] = neg_omg;
src[1] = omega_dot;
src[2] = -rotating_U_dot;
src[3] = rotating_V_dot;

</%pyfr:macro>
