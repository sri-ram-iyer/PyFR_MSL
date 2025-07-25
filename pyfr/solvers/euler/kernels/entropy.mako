<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='compute_entropy' params='u, d, p, e, rote' externs='t, omg_sqr, global_U, global_V'>
    d = u[0];
    fpdtype_t rcpd = 1.0/d;
    fpdtype_t E = u[${nvars - 1}];

    // Compute the pressure
    p = ${c['gamma'] - 1}*(E - 0.5*rcpd*(${pyfr.dot('u[{i}]', i=(1, ndims + 1))}) + d*rote*omg_sqr);

    // Compute numerical or specific physical entropy
    % if e_func == 'numerical':
    e = (d > 0 && p > 0) ? d*(log(p) - ${c['gamma']}*log(d)) : ${fpdtype_max};
    % elif e_func == 'physical':
    e = (d > 0 && p > 0) ? p*pow(rcpd, ${c['gamma']}) : ${fpdtype_max};
    % endif
</%pyfr:macro>
