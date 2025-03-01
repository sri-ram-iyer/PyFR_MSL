<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='source' params='t, u, ploc, src' externs='omega_dot, neg_omg'>

src[0] = neg_omg;
src[1] = omega_dot;

</%pyfr:macro>
