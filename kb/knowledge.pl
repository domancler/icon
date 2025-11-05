% kb/knowledge.pl

% Fatti statici: team/1, city/1, stadium/1, plays_in/2, has_stadium/2, rivals/2, home_advantage/2

:- dynamic points_last5/2.
:- dynamic gd_last5/2.
:- dynamic match/3.

is_derby(H, A) :- plays_in(H, C), plays_in(A, C), !.
is_derby(H, A) :- rivals(H, A) ; rivals(A, H).

strong_form(T) :- points_last5(T, P), P >= 8.
strong_form(T) :- gd_last5(T, GD), GD >= 6.

high_gap(Home, Away) :-
    gd_last5(Home, GDH), gd_last5(Away, GDA),
    Gap is GDH - GDA, Gap >= 6.

home_edge(Home) :- home_advantage(Home, A), A >= 0.10.

likely_draw(Home, Away) :- is_derby(Home, Away).
likely_home(Home, Away) :- high_gap(Home, Away).
likely_home(Home, _) :- strong_form(Home).
likely_home(Home, _) :- home_edge(Home).
likely_away(Home, Away) :- high_gap(Away, Home).
likely_away(_, Away) :- strong_form(Away).

% --- team ---
team(atalanta). team(benevento). team(bologna). team(cagliari).
team(como). team(cremonese). team(crotone). team(empoli).
team(fiorentina). team(frosinone). team(genoa). team(inter).
team(juventus). team(lazio). team(lecce). team(milan).
team(monza). team(napoli). team(parma). team(roma).
team(salernitana). team(sampdoria). team(sassuolo). team(spezia).
team(torino). team(udinese). team(venezia). team(verona).

% --- city/1 ---
city(bergamo). city(benevento). city(bologna).
city(cagliari). city(como). city(cremona).
city(crotone). city(empoli). city(firenze).
city(frosinone). city(genova). city(milano).
city(torino). city(roma). city(lecce).
city(monza). city(napoli). city(parma).
city(salerno). city(reggio_emilia). city(la_spezia).
city(udine). city(venezia). city(verona).

% --- stadium/1 ---
stadium(gewiss_stadium). stadium(stadio_ciro_vigorito).
stadium(stadio_renato_dall_ara). stadium(unipol_domus).
stadium(stadio_giuseppe_sinigaglia). stadium(stadio_giovanni_zini).
stadium(stadio_ezio_scida). stadium(stadio_carlo_castellani).
stadium(stadio_artemio_franchi). stadium(stadio_benito_stirpe).
stadium(stadio_luigi_ferraris). stadium(stadio_giuseppe_meazza).
stadium(allianz_stadium). stadium(stadio_olimpico).
stadium(stadio_via_del_mare). stadium(u_power_stadium).
stadium(stadio_diego_armando_maradona). stadium(stadio_ennio_tardini).
stadium(stadio_arechi). stadium(mapei_stadium_citta_del_tricolore).
stadium(stadio_alberto_picco). stadium(stadio_olimpico_grande_torino).
stadium(bluenergy_stadium). stadium(stadio_pier_luigi_penzo).
stadium(stadio_marcantonio_bentegodi).

% --- plays_in/2 ---
plays_in(atalanta, bergamo). plays_in(benevento, benevento).
plays_in(bologna, bologna). plays_in(cagliari, cagliari).
plays_in(como, como). plays_in(cremonese, cremona).
plays_in(crotone, crotone). plays_in(empoli, empoli).
plays_in(fiorentina, firenze). plays_in(frosinone, frosinone).
plays_in(genoa, genova). plays_in(inter, milano).
plays_in(juventus, torino). plays_in(lazio, roma).
plays_in(lecce, lecce). plays_in(milan, milano).
plays_in(monza, monza). plays_in(napoli, napoli).
plays_in(parma, parma). plays_in(roma, roma).
plays_in(salernitana, salerno). plays_in(sampdoria, genova).
plays_in(sassuolo, reggio_emilia). plays_in(spezia, la_spezia).
plays_in(torino, torino). plays_in(udinese, udine).
plays_in(venezia, venezia). plays_in(verona, verona).

% --- has_stadium/2 ---
has_stadium(atalanta, gewiss_stadium).
has_stadium(benevento, stadio_ciro_vigorito).
has_stadium(bologna, stadio_renato_dall_ara).
has_stadium(cagliari, unipol_domus).
has_stadium(como, stadio_giuseppe_sinigaglia).
has_stadium(cremonese, stadio_giovanni_zini).
has_stadium(crotone, stadio_ezio_scida).
has_stadium(empoli, stadio_carlo_castellani).
has_stadium(fiorentina, stadio_artemio_franchi).
has_stadium(frosinone, stadio_benito_stirpe).
has_stadium(genoa, stadio_luigi_ferraris).
has_stadium(inter, stadio_giuseppe_meazza).
has_stadium(juventus, allianz_stadium).
has_stadium(lazio, stadio_olimpico).
has_stadium(lecce, stadio_via_del_mare).
has_stadium(milan, stadio_giuseppe_meazza).
has_stadium(monza, u_power_stadium).
has_stadium(napoli, stadio_diego_armando_maradona).
has_stadium(parma, stadio_ennio_tardini).
has_stadium(roma, stadio_olimpico).
has_stadium(salernitana, stadio_arechi).
has_stadium(sampdoria, stadio_luigi_ferraris).
has_stadium(sassuolo, mapei_stadium_citta_del_tricolore).
has_stadium(spezia, stadio_alberto_picco).
has_stadium(torino, stadio_olimpico_grande_torino).
has_stadium(udinese, bluenergy_stadium).
has_stadium(venezia, stadio_pier_luigi_penzo).
has_stadium(verona, stadio_marcantonio_bentegodi).

% --- home_advantage/2 ---
home_advantage(atalanta, 0.085). home_advantage(benevento, 0.05).
home_advantage(bologna, 0.162). home_advantage(cagliari, 0.133).
home_advantage(como, 0.157). home_advantage(cremonese, 0.15).
home_advantage(crotone, 0.167). home_advantage(empoli, 0.114).
home_advantage(fiorentina, 0.172). home_advantage(frosinone, 0.2).
home_advantage(genoa, 0.128). home_advantage(inter, 0.151).
home_advantage(juventus, 0.155). home_advantage(lazio, 0.118).
home_advantage(lecce, 0.116). home_advantage(milan, 0.092).
home_advantage(monza, 0.099). home_advantage(napoli, 0.102).
home_advantage(parma, 0.141). home_advantage(roma, 0.189).
home_advantage(salernitana, 0.122). home_advantage(sampdoria, 0.122).
home_advantage(sassuolo, 0.112). home_advantage(spezia, 0.117).
home_advantage(torino, 0.111). home_advantage(udinese, 0.101).
home_advantage(venezia, 0.143). home_advantage(verona, 0.136).

% --- rivals/2 ---
rivals(inter, milan).
rivals(lazio, roma).
rivals(juventus, torino).
rivals(genoa, sampdoria).