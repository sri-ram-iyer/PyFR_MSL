<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='bc_rsolve_state' params='ul, nl, ur, rote' externs='ploc, t'>
% for i in range(nvars):
    ur[${i}] = ul[${i}];
% endfor
</%pyfr:macro>
