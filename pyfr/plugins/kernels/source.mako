<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='source' params='t, u, ploc, src' externs='omega_dot, neg_omg, global_U_dot, global_V_dot'>

src[0] = neg_omg;
src[1] = omega_dot;
src[2] = -global_U_dot;
src[3] = global_V_dot;

</%pyfr:macro>
