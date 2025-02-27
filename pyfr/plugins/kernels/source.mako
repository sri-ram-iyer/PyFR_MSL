<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='source' params='t, u, ploc, src' externs='omg_sqr, neg_omg'>
% for i, ex in enumerate(src_exprs):
    src[0] += neg_omg;
    src[1] += omega_dot;
% endfor
</%pyfr:macro>
