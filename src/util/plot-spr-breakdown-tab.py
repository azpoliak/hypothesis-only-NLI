import itertools
import pdb
import argparse

PROTO_ROLES = set(["sentient", "aware of being involved", "existed after", "existed before", "existed during", "chose to be involved", "changed possession", "describes the location", "stationary during", "made physical contact with someone or something", "used in", "caused a change", "changes location during", "altered or somehow changed during", "existed as a physical object", "caused the", "used in carrying out"])

role2str = {"sentient" : "sentient", "aware of being involved" : "aware", "existed after" : "existed after", \
            "existed before" : "existed before", "existed during" : "existed during", "chose to be involved" : "volitional", \
            "changed possession" : "chang. possession", "describes the location": "location", "stationary during" : "stationary during" , \
            "made physical contact with someone or something" : "physical contact", "caused a change" : "changed", \
            "changes location during" : "moved", "altered or somehow changed during" : "chang. state", \
            "existed as a physical object" : "physically existed", "caused the" : "caused", "used in carrying out" : "used in"}


AGENT_ROLES = ["existed as a physical object", "sentient", "aware of being involved", "chose to be involved", \
                "existed before", "caused the"]

PATIENT_ROLES = [key for key in role2str if key not in AGENT_ROLES]

def get_data(args):
 lbls_file = open(args.gold)
 src_file = open(args.src)
 pred_file = open(args.pred)

 idx = {}
 for i, trip in enumerate(itertools.izip(lbls_file, src_file, pred_file)):
   idx[i] = trip #trip[0], trip[1], trip[2], trip[3], trip[4]
 
 return idx

def get_idx_by_role(data):
  role2idx = {}
  for role in PROTO_ROLES:
    role2idx[role] = []
  for idx in data:
    found_role = False
    hyp = data[idx][1].split("|||")[-1]
    for role in PROTO_ROLES:
      if role in hyp:
        found_role = True
        role2idx[role].append(idx)
    if not found_role:
      print hyp    
  return role2idx

def main(args):
  data = get_data(args) #args.src, args.gold, args.pred)

  maj_entailed = 0.0

  role2idx = get_idx_by_role(data)

  role2pos_count = {}
  print "\small{%s}\t& \small{%s}\t& \small{%s}\t& \small{\\%% Change} \\\\ \\hline" % ("Proto-Role", "Hypothesis-model", "MAJ")
  for role in  AGENT_ROLES + PATIENT_ROLES:
    locs = role2idx[role]
    if role not in role2pos_count:
      role2pos_count[role] = 0.0
    role_tot, corr = 0, 0
    for loc in locs:
      if "not" not in data[loc][0]:
        role2pos_count[role] += 1
      if data[loc][0] == data[loc][2]:
        corr += 1
      role_tot += 1.0
    maj = 100*(max(1 - (role2pos_count[role]/role_tot), (role2pos_count[role]/role_tot)))
    hyp_mod = 100*corr/role_tot
    print "\small{%s}\t& %.2f\t& %.2f & %.2f\\%% \\\\" % (role2str[role], hyp_mod, maj, 100 * ((hyp_mod/maj) - 1))
    #print "\small{%s}\t& %.2f\t& %.2f & %.2f\%  \\\\" % (role2str[role], 100*corr/role_tot, 100*(max(1 - (role2pos_count[role]/role_tot), (role2pos_count[role]/role_tot))), ((100*corr/role_tot) / 100*(max(1 - (role2pos_count[role]/
    #role_tot), (role2pos_count[role]/role_tot)))) - 100   )
    if 1 - (role2pos_count[role]/role_tot) <  (role2pos_count[role]/role_tot):
      maj_entailed += 1
  print "For %.2f percent of the roles, the majority label was entailed." % (100 * maj_entailed / len(role2idx))


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Analysis on sprl.')
  parser.add_argument('--src', help="path to source file")
  parser.add_argument('--gold', help="path to gold labels file")
  parser.add_argument('--pred', help="path to pred labels file")

  args = parser.parse_args()

  main(args)

